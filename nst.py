import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from PIL import Image

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
    model.eval()

    content_layers = [7]
    style_layers = [0, 1, 2, 3, 4, 5, 6]

    content_image_name = sys.argv[1]
    content_image_path = os.path.abspath(content_image_name)
    print(content_image_path)
    content_image = Image.open(content_image_path)

    style_image_name = sys.argv[2]
    style_image_path = os.path.abspath(style_image_name)
    print(style_image_path)
    style_image = Image.open(style_image_path)
    exit()

    width = 200
    height = 200
    x = torch.rand(size=(1, 3, width, height), dtype=torch.float32, device=device, requires_grad=True)
    x_content_features, x_style_features = get_features(x, model, content_layers, style_layers)

    y = model(x)
    L = torch.mean(y ** 2)
    L.backward()
    print(x.grad.shape)
    exit()

    plt.imshow(x.detach().numpy())
    plt.show()

    optimizer = optim.Adam(x, lr=3e-4)


if __name__ == "__main__":
    main()