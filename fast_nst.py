import sys
import os
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import vgg16, VGG16_Weights
from torchvision.io import read_image
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.transforms.functional import resize

import random
random.seed(42069)
np.random.seed(42069)
torch.manual_seed(42069)
torch.cuda.manual_seed(42069)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def gram(F):
    return torch.matmul(F, F.transpose(1, 2))


class Model(nn.Module):

    def __init__(self, pretrained_model, preprocess, content_layers, style_layers):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.preprocess = preprocess
        self.content_layers = content_layers
        self.style_layers = style_layers

        for i, layer in enumerate(self.pretrained_model):
            if type(layer) == nn.MaxPool2d:
                self.pretrained_model[i] = nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding, ceil_mode=layer.ceil_mode)

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        content_features = []
        style_features = []

        batch_size = x.shape[0]

        x = self.preprocess(x)

        for i, layer in enumerate(self.pretrained_model):
            x = layer(x)
            if i in self.content_layers:
                F = torch.reshape(x, (batch_size, x.shape[1], -1))
                content_features.append(F)
            if i in self.style_layers:
                F = torch.reshape(x, (batch_size, x.shape[1], -1))
                style_features.append(F)

        return content_features, style_features


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_relu=True):
        super().__init__()

        if stride >= 1:
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)]
        else:
            stride = int(1 / stride)
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=stride - 1)]

        layers.append(nn.InstanceNorm2d(num_features=out_channels, affine=True))

        if use_relu:
            layers.append(nn.ReLU())

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels),
            ConvBlock(channels, channels, use_relu=False)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x + self.block(x)
        x = self.relu(x)
        return x


class Transform(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            ConvBlock(3, 50, kernel_size=9, stride=2, padding=4),
            ConvBlock(50, 100, stride=2),
            ResBlock(100),
            ResBlock(100),
            ResBlock(100),
            ResBlock(100),
            ResBlock(100),
            ConvBlock(100, 50, kernel_size=3, stride=0.5),
            ConvBlock(50, 3, kernel_size=9, stride=0.5, padding=4, use_relu=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.network(x)
        return x


def train():
    image_height = 256
    image_width = 256

    style_image_name = sys.argv[1]
    style_image_path = os.path.abspath(style_image_name)
    style_image = read_image(style_image_path)
    style_image = resize(style_image, (image_height, image_width))
    style_image = style_image.unsqueeze(0)
    style_image = style_image.to(device)

    pretrained_weights = VGG16_Weights.DEFAULT
    pretrained_model = vgg16(weights=pretrained_weights).features

    preprocess = pretrained_weights.transforms()
    preprocess.crop_size = [image_height, image_width]
    preprocess.resize_size = [image_height, image_width]

    content_layers = [15]
    # content_layers = [8, 15, 22, 25, 29]
    style_layers = [3, 8, 15, 22]

    model = Model(pretrained_model, preprocess, content_layers, style_layers)
    model.to(device)

    _, style_features = model(style_image)
    for i in range(len(style_layers)):
        style_features[i] = gram(style_features[i])

    transform = Transform()
    transform.to(device)

    optimizer = Adam(transform.parameters(), lr=1e-3)

    data_transform = Compose([Resize((image_height, image_width)), ToTensor()])
    dataset = ImageFolder(root="~/Downloads/MSCOCO/", transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=5)

    content_weight = 1
    style_weight = 1e9
    total_variation_weight = 1e3

    style = style_image_name.split(".")[0]
    transform_path = os.path.join(os.getcwd(), "Models")
    if not os.path.isdir(transform_path):
        os.mkdir(transform_path)
    style_transform_path = os.path.join(transform_path, style + ".pt")

    try:
        with open(os.path.join(transform_path, "progress_" + style), "r") as f:
            e, i = f.readlines()
            epoch_start = int(e)
            iteration_start = int(i)

            transform.load_state_dict(torch.load(style_transform_path))
            transform.train()
    except FileNotFoundError:
        epoch_start = 1
        iteration_start = 0

    epochs = 2
    for epoch in range(epoch_start, epochs + 1):
        try:
            for iteration, (batch, _) in enumerate(dataloader, iteration_start):
                optimizer.zero_grad()

                batch = batch.to(device)
                content_features, _ = model(batch)

                x = transform(batch)
                x_content_features, x_style_features = model(x)

                L_content = 0
                for i in range(len(content_layers)):
                    L_content += torch.sum((x_content_features[i] - content_features[i]) ** 2) / 2
                L_content /= len(content_layers)

                L_style = 0
                for i in range(len(style_layers)):
                    N, M = x_style_features[i].shape[1:]
                    L_style += torch.sum((gram(x_style_features[i]) - style_features[i]) ** 2) / (4 * N**2 * M**2)
                L_style /= len(style_layers)

                # Total variation loss
                high_pass_height = x[:, :, 1:, :] - x[:, :, :-1, :]
                high_pass_width = x[:, :, :, 1:] - x[:, :, :, :-1]
                total_variation_height = torch.mean(torch.abs(high_pass_height))
                total_variation_width = torch.mean(torch.abs(high_pass_width))
                L_total_variation = total_variation_height + total_variation_width

                L = L_content * content_weight + L_style * style_weight + L_total_variation * total_variation_weight
                L.backward()

                optimizer.step()

                print(f"Epoch: {epoch:2d} | Iteration: {iteration:5,d} | Total Loss: {L.item():14,.3f}".replace(",", " "))

                if iteration % 100 == 0:
                    torch.save(transform.state_dict(), style_transform_path)
                    with open(os.path.join(transform_path, "progress_" + style), "w") as f:
                        f.write(str(epoch) + "\n")
                        f.write(str(iteration))

            iteration_start = 0

        except KeyboardInterrupt:
            torch.save(transform.state_dict(), style_transform_path)
            with open(os.path.join(transform_path, "progress_" + style), "w") as f:
                f.write(str(epoch) + "\n")
                f.write(str(iteration))

            sys.exit()

    torch.save(transform.state_dict(), style_transform_path)
    with open(os.path.join(transform_path, "progress_" + style), "w") as f:
        f.write(str(1) + "\n")
        f.write(str(0))


def apply():
    content_image_name = sys.argv[1]
    content_image_path = os.path.abspath(content_image_name)
    content_image = read_image(content_image_path).float() / 255
    content_image = content_image.to(device)

    style_image_name = sys.argv[2]
    style = style_image_name.split(".")[0]
    style_transform_path = os.path.join(os.getcwd(), f"Models/{style}.pt")

    transform = Transform()
    try:
        transform.load_state_dict(torch.load(style_transform_path))
    except FileNotFoundError:
        print("Trained model not found. Exiting")
        sys.exit()

    # transform.eval() # Keep instance normalization also when aplying the network
    transform = transform.to(device)

    output = transform(content_image)
    output_image = output.detach().cpu()

    output_file_path = os.path.join(os.getcwd(), "FastNSTOutput")
    if not os.path.isdir(output_file_path):
        os.mkdir(output_file_path)

    content = content_image_name.split(".")[0]

    fp = os.path.join(output_file_path, f"{content}_{style}.jpg")
    save_image(output_image, fp, quality=75)


def main():
    if len(sys.argv) == 2:
        train()
    elif len(sys.argv) == 3:
        apply()
    else:
        print("Argument error.")
        print("To train on a new style:")
        print(f"python {sys.argv[0]} <style_image>")
        print()
        print("To apply a trained style to an image:")
        print(f"python {sys.argv[0]} <content_image> <style_image>")


if __name__ == "__main__":
    main()
