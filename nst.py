import sys
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
# from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models import vgg19, VGG19_Weights
from torchvision.io import read_image

import random
random.seed(42069)
np.random.seed(42069)
torch.manual_seed(42069)


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"


def get_features(x, model, preprocess, content_layers, style_layers):
    content_features = []
    style_features = []

    preprocess.crop_size = x.shape[1:]
    preprocess.resize_size = x.shape[1:]
    x = preprocess(x).unsqueeze(0)

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

    # weights = EfficientNet_V2_S_Weights.DEFAULT
    # model = efficientnet_v2_s(weights=weights)
    weights = VGG19_Weights.DEFAULT
    model = vgg19(weights=weights)
    model.to(device)
    model.eval()

    # for i in range(len(model.features)):
        # if type(model.features[i]) == torch.nn.modules.conv.Conv2d:
            # print(i)
        # print(type(model.features[i]))
    # exit()

    preprocess = weights.transforms()

    content_layers = [0]
    style_layers = [0]

    content_weight = 1
    style_weight = 1000

    content_image_name = sys.argv[1]
    content_image_path = os.path.abspath(content_image_name)
    content_image = read_image(content_image_path)
    content_image = content_image.to(device)

    with torch.no_grad():
        content_features, _ = get_features(content_image, model, preprocess, content_layers, style_layers)

    style_image_name = sys.argv[1]
    style_image_path = os.path.abspath(style_image_name)
    style_image = read_image(style_image_path)
    style_image = style_image.to(device)

    with torch.no_grad():
        _, style_features = get_features(style_image, model, preprocess, style_layers, style_layers)

    x = torch.rand(size=content_image.shape, dtype=torch.float32, device=device, requires_grad=True)
    x = x.to(device)

    optimizer = optim.LBFGS([x])

    plt.ion()

    epochs = 2

    content_losses = []
    style_losses = []

    for epoch in range(1, epochs + 1):

        def closure(epoch):
            with torch.no_grad():
                x.clamp_(0, 1)

            optimizer.zero_grad()

            image = x.detach().cpu().numpy().transpose(1, 2, 0)
            plt.imshow(image)
            plt.pause(0.001)

            x_content_features, x_style_features = get_features(x, model, preprocess, content_layers, style_layers)

            L_content = 0
            for i in range(len(content_layers)):
                L_content += 0.5 * torch.sum((x_content_features[i] - content_features[i]) ** 2)

            L_style = 0

            L = L_content * content_weight + L_style * style_weight
            L.backward()

            content_losses.append(L_content)
            style_losses.append(0)

            return L_content

        optimizer.step(lambda: closure(epoch))

        print(f"Epoch: {epoch}, Content Loss: {content_losses[-1]}")

        # with torch.no_grad():
            # x.clamp_(0, 1)


if __name__ == "__main__":
    main()
