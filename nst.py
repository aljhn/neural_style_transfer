import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg19, VGG19_Weights
from torchvision.io import read_image
from torchvision.utils import save_image
from torchvision.transforms.functional import resize

import random
random.seed(42069)
np.random.seed(42069)
torch.manual_seed(42069)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def gram(F):
    return torch.matmul(F, F.t())


class Model(nn.Module):

    def __init__(self, pretrained_model, preprocess, content_layers, style_layers):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.preprocess = preprocess
        self.content_layers = content_layers
        self.style_layers = style_layers

        # Replace MaxPool with AveragePool by paper recommendation
        for i, layer in enumerate(self.pretrained_model):
            if type(layer) == nn.MaxPool2d:
                self.pretrained_model[i] = nn.AvgPool2d(kernel_size=layer.kernel_size, stride=layer.stride, padding=layer.padding, ceil_mode=layer.ceil_mode)

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        content_features = []
        style_features = []

        x = self.preprocess(x).unsqueeze(0)

        for i, layer in enumerate(self.pretrained_model):
            x = layer(x)
            if i in self.content_layers:
                F = torch.reshape(x, (x.shape[1], -1))
                content_features.append(F)
            if i in self.style_layers:
                F = torch.reshape(x, (x.shape[1], -1))
                style_features.append(F)

        return content_features, style_features


def main():
    if len(sys.argv) < 3:
        print("Argument error. Run as:")
        print(f"python {sys.argv[0]} <content_image> <style_image>")
        sys.exit()

    pretrained_weights = VGG19_Weights.DEFAULT
    pretrained_model = vgg19(weights=pretrained_weights).features

    image_width = 400
    image_height = 400

    preprocess = pretrained_weights.transforms()

    # Default image size is 224x224, so change manually
    preprocess.crop_size = [image_height, image_width]
    preprocess.resize_size = [image_height, image_width]

    content_layers = [21]
    style_layers = [0, 5, 10, 19, 28]

    content_weight = 1
    style_weight = 1e10
    total_variation_weight = 1e2

    model = Model(pretrained_model, preprocess, content_layers, style_layers)
    model.to(device)

    content_image_name = sys.argv[1]
    content_image_path = os.path.abspath(content_image_name)
    content_image = read_image(content_image_path)
    content_image_height, content_image_width = content_image.shape[1:]
    content_image = resize(content_image, (image_height, image_width))
    content_image = content_image.to(device)

    with torch.no_grad():
        content_features, _ = model(content_image)

    style_image_name = sys.argv[2]
    style_image_path = os.path.abspath(style_image_name)
    style_image = read_image(style_image_path)
    style_image = resize(style_image, (image_height, image_width))
    style_image = style_image.to(device)

    with torch.no_grad():
        _, style_features = model(style_image)
        for i in range(len(style_layers)):
            style_features[i] = gram(style_features[i])

    # x = torch.rand(size=(3, image_height, image_width), dtype=torch.float32, device=device, requires_grad=True)
    x = torch.clone(content_image).float() / 255
    x.requires_grad = True
    x = x.to(device)

    optimizer = optim.LBFGS([x])

    plt.ion()

    content_losses = []
    style_losses = []
    tv_losses = []
    total_losses = []

    output_file_path = os.path.join(os.getcwd(), "NSTOutput")
    if not os.path.isdir(output_file_path):
        os.mkdir(output_file_path)

    def closure():
        with torch.no_grad():
            x.clamp_(0, 1)

        image = x.detach().cpu()
        if len(total_losses) % 10 == 0:
            fp = os.path.join(output_file_path, f"image{len(total_losses)}.jpg")
            output_image = resize(image, (content_image_height, content_image_width))
            save_image(output_image, fp, quality=75)

        # image = image.numpy().transpose(1, 2, 0)
        # plt.imshow(image)
        # plt.pause(0.001)

        optimizer.zero_grad()

        x_content_features, x_style_features = model(x)

        L_content = 0
        for i in range(len(content_layers)):
            L_content += torch.sum((x_content_features[i] - content_features[i]) ** 2) / 2
        L_content /= len(content_layers)

        L_style = 0
        for i in range(len(style_layers)):
            N, M = x_style_features[i].shape
            L_style += torch.sum((gram(x_style_features[i]) - style_features[i]) ** 2) / (4 * N**2 * M**2)
        L_style /= len(style_layers)

        # Total variation loss
        high_pass_height = x[:, 1:, :] - x[:, :-1, :]
        high_pass_width = x[:, :, 1:] - x[:, :, :-1]
        total_variation_height = torch.mean(torch.abs(high_pass_height))
        total_variation_width = torch.mean(torch.abs(high_pass_width))
        L_total_variation = total_variation_height + total_variation_width

        L = L_content * content_weight + L_style * style_weight + L_total_variation * total_variation_weight
        L.backward()

        content_losses.append(L_content)
        style_losses.append(L_style)
        tv_losses.append(L_total_variation)
        total_losses.append(L)

        print(f"Iteration: {len(total_losses):3d} | Total Loss: {total_losses[-1]:14,.3f}".replace(",", " "))

        return L

    while True:
        try:
            optimizer.step(closure)

        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    main()
