import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.optim as optim
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
import torchvision.transforms as transforms

import random
random.seed(42069)
np.random.seed(42069)
torch.manual_seed(42069)


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"


def get_features(x, model, content_layers, style_layers):
    content_features = []
    style_features = []

    for layer in range(len(model.features)):
        x = model.features[layer](x)
        if layer in content_layers:
            content_features.append(torch.reshape(x, (x.shape[1], -1)))
        elif layer in style_layers:
            style_features.append(torch.reshape(x, (x.shape[1], -1)))

    return content_features, style_features


def main():
    if len(sys.argv) < 3:
        print("Argument error. Run as:")
        print(f"python {sys.argv[0]} <content_image> <style_image>")
        sys.exit()

    weights = EfficientNet_V2_S_Weights.DEFAULT
    model = efficientnet_v2_s(weights=weights)
    model.to(device)
    model.eval()

    content_layers = [7]
    style_layers = [0, 1, 2, 3, 4, 5, 6]

    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float32)
    ])

    content_image_name = sys.argv[1]
    content_image_path = os.path.abspath(content_image_name)
    content_image = Image.open(content_image_path)
    content_image = transform(content_image)
    content_image = content_image.unsqueeze(0)
    content_image.to(device)
    with torch.no_grad():
        content_content_features, content_style_features = get_features(content_image, model, content_layers, style_layers)

    # style_image_name = sys.argv[2]
    # style_image_path = os.path.abspath(style_image_name)
    # style_image = Image.open(style_image_path)
    # style_image = style_image.resize(content_image.shape[2:])
    # style_image = transform(style_image)
    # style_image = style_image.unsqueeze(0)
    # style_image.to(device)
    # with torch.no_grad():
        # style_content_features, style_style_features = get_features(style_image, model, content_layers, style_layers)

    height = content_image.shape[2]
    width = content_image.shape[3]
    x = torch.rand(size=(1, 3, height, width), dtype=torch.float32, device=device, requires_grad=True)

    optimizer = optim.Adam([x], lr=1e-1)

    plt.ion()

    epochs = 100
    for epoch in range(1, epochs + 1):
        image = x.detach().numpy()[0, :, :, :].transpose(1, 2, 0)
        image = np.clip(image, 0, 1)
        plt.imshow(image)
        plt.pause(0.001)

        x_content_features, x_style_features = get_features(x, model, content_layers, style_layers)

        L_content = 0
        for i in range(len(content_layers)):
            L_content += 0.5 * torch.sum((x_content_features[i] - content_content_features[i]) ** 2)

        L_content.backward()
        optimizer.step()

        print(f"Epoch: {epoch}, Content Loss: {L_content}")


if __name__ == "__main__":
    main()
