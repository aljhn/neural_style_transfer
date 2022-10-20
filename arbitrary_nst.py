import sys
import os
import numpy as np
import copy

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import vgg19, VGG19_Weights
from torchvision.io import read_image
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, RandomCrop, ToTensor

import random
random.seed(42069)
np.random.seed(42069)
torch.manual_seed(42069)
torch.cuda.manual_seed(42069)

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AdaIN(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, c, s):
        c_std, c_mean = torch.std_mean(input=c, dim=(2, 3), unbiased=True, keepdim=True)
        s_std, s_mean = torch.std_mean(input=s, dim=(2, 3), unbiased=True, keepdim=True)
        return s_std * (c - c_mean) / c_std + s_mean


class Encoder(nn.Module):

    def __init__(self, pretrained_model, preprocess, final_layer, style_layers):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.preprocess = preprocess
        self.final_layer = final_layer
        self.style_layers = style_layers

        for layer in self.pretrained_model:
            if type(layer) == nn.modules.conv.Conv2d:
                layer.padding_mode = "reflect"

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

    def forward(self, x):
        style_features = []

        x = self.preprocess(x)

        for i, layer in enumerate(self.pretrained_model):
            x = layer(x)

            if i in self.style_layers:
                style_features.append(x)

            if i == self.final_layer:
                break

        return x, style_features


class Decoder(nn.Module):

    def __init__(self, encoder):
        super().__init__()
        layers = []
        for i, layer in enumerate(encoder.pretrained_model):
            if type(layer) == nn.modules.conv.Conv2d:
                layer = nn.Conv2d(
                    in_channels=layer.out_channels,
                    out_channels=layer.in_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    padding_mode=layer.padding_mode
                )
            elif type(layer) == nn.modules.MaxPool2d:
                layer = nn.Upsample(scale_factor=2, mode="nearest")

            layers.append(copy.deepcopy(layer))

            if i == encoder.final_layer:
                break

        self.network = nn.Sequential(*reversed(layers), nn.Sigmoid())
        # self.network = nn.Sequential(*reversed(layers))

    def forward(self, x):
        return self.network(x)


def train():
    image_size = 256

    pretrained_weights = VGG19_Weights.DEFAULT
    pretrained_model = vgg19(weights=pretrained_weights).features

    preprocess = pretrained_weights.transforms()
    preprocess.crop_size = [image_size, image_size]
    preprocess.resize_size = [image_size, image_size]

    final_layer = 19
    style_layers = [1, 6, 10, 19]

    encoder = Encoder(pretrained_model, preprocess, final_layer, style_layers)
    encoder.to(device)

    decoder = Decoder(encoder)
    decoder.to(device)

    adain = AdaIN()
    adain.to(device)

    optimizer = Adam(decoder.parameters(), lr=1e-5)

    batch_size = 5

    data_transform = Compose([Resize(image_size * 2), RandomCrop(image_size), ToTensor()])
    # dataset_content = ImageFolder(root="/datasets/MSCOCO/", transform=data_transform)
    dataset_content = ImageFolder(root="/datasets/PascalVOC/", transform=data_transform)
    #dataset_style = ImageFolder(root="~/datasets/WikiArt/", transform=data_transform)
    dataset_style = ImageFolder(root="/datasets/BestArtworksOfAllTime/", transform=data_transform)
    dataloader_content = DataLoader(dataset_content, batch_size=batch_size)
    dataloader_style = DataLoader(dataset_style, batch_size=batch_size, shuffle=True)
    dataloader_style_iterator = iter(dataloader_style)

    path = os.path.join(os.getcwd(), "Models")
    if not os.path.isdir(path):
        os.mkdir(path)
    progress_path = os.path.join(path, "progress_arbitrary")
    model_path = os.path.join(path, "arbitrary.pt")

    try:
        with open(progress_path, "r") as f:
            e, i = f.readlines()
            epoch_start = int(e)
            iteration_start = int(i)

        decoder.load_state_dict(torch.load(model_path))
        decoder.train()
    except FileNotFoundError:
        epoch_start = 1
        iteration_start = 0

    style_weight = 1e3

    mse = nn.MSELoss()
    
    epochs = 50
    for epoch in range(epoch_start, epochs + 1):
        try:
            for iteration, (batch_content, _) in enumerate(dataloader_content, iteration_start):
                batch_style, _ = next(dataloader_style_iterator)

                batch_style = batch_style.to(device)
                batch_content = batch_content.to(device)
                
                optimizer.zero_grad()

                f_c, _ = encoder(batch_content)
                f_s, f_s_styles = encoder(batch_style)

                t = adain(f_c, f_s)
                if torch.isnan(t).any():
                    continue

                g_t = decoder(t)
                
                f_g_t, f_g_t_styles = encoder(g_t)

                L_content = mse(f_g_t, t)

                L_style = 0
                for i in range(len(style_layers)):
                    f_s_std, f_s_mean = torch.std_mean(input=f_s_styles[i], dim=(2, 3), unbiased=True)
                    f_g_t_std, f_g_t_mean = torch.std_mean(input=f_g_t_styles[i], dim=(2, 3), unbiased=True)
                    L_style += mse(f_s_mean, f_g_t_mean) + mse(f_s_std, f_g_t_std)
                L_style /= len(style_layers)

                L = L_content + style_weight * L_style
                L.backward()

                optimizer.step()
                
                print(f"Epoch: {epoch:2d} | Iteration: {iteration:5,d} | Total Loss: {L.item():14,.3f}".replace(",", " "))

                if iteration % 100 == 0:
                    torch.save(decoder.state_dict(), model_path)
                    with open(progress_path, "w") as f:
                        f.write(str(epoch) + "\n")
                        f.write(str(iteration))

            iteration_start = 0

        except KeyboardInterrupt:
            torch.save(decoder.state_dict(), model_path)
            with open(progress_path, "w") as f:
                f.write(str(epoch) + "\n")
                f.write(str(iteration))

            sys.exit()
        
        except StopIteration:
            iteration_start = 0
            dataloader_style_iterator = iter(dataloader_style)

    torch.save(decoder.state_dict(), model_path)
    with open(progress_path, "w") as f:
        f.write(str(1) + "\n")
        f.write(str(0))


def apply():
    content_image_name = sys.argv[1]
    content_image_path = os.path.abspath(content_image_name)
    content_image = read_image(content_image_path).float() / 255
    content_image = content_image.unsqueeze(0)
    content_image = content_image.to(device)

    image_height, image_width = content_image.shape[2:4]

    style_image_name = sys.argv[2]
    style_image_path = os.path.abspath(style_image_name)
    style_image = read_image(style_image_path).float() / 255
    style_image = style_image.unsqueeze(0)
    style_image = style_image.to(device)

    pretrained_weights = VGG19_Weights.DEFAULT
    pretrained_model = vgg19(weights=pretrained_weights).features

    preprocess = pretrained_weights.transforms()
    preprocess.crop_size = [image_height, image_width]
    preprocess.resize_size = [image_height, image_width]

    final_layer = 19
    style_layers = [1, 6, 10, 19]

    encoder = Encoder(pretrained_model, preprocess, final_layer, style_layers)
    encoder.to(device)

    decoder = Decoder(encoder)
    decoder.to(device)

    adain = AdaIN()
    adain.to(device)

    path = os.path.join(os.getcwd(), "Models")
    if not os.path.isdir(path):
        os.mkdir(path)
    model_path = os.path.join(path, "arbitrary.pt")

    try:
        decoder.load_state_dict(torch.load(model_path))
    except FileNotFoundError:
        pass

    with torch.no_grad():
        f_c, _ = encoder(content_image)
        f_s, _ = encoder(style_image)
        t = adain(f_c, f_s)
        g_t = decoder(t)

    output_image = g_t.cpu().squeeze(0)

    output_file_path = os.path.join(os.getcwd(), "ArbitraryNSTOutput")
    if not os.path.isdir(output_file_path):
        os.mkdir(output_file_path)

    content = content_image_name.split(".")[0]
    style = style_image_name.split(".")[0]

    fp = os.path.join(output_file_path, f"{content}_{style}.jpg")
    save_image(output_image, fp)


def main():
    if len(sys.argv) == 1:
        train()
    elif len(sys.argv) == 3:
        apply()
    else:
        print("Argument error.")
        print("Train by running without arguments:")
        print(f"python {sys.argv[0]}")
        print()
        print("Apply a style image to a content image:")
        print(f"python {sys.argv[0]} <content_image> <style_image>")


if __name__ == "__main__":
    main()
