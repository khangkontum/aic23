from sklearn import preprocessing
import torch
import clip
from PIL import Image
import numpy as np

import os
from PIL import Image
from tqdm import tqdm

input_folder = "./raw/frames/"

output_folder = "./raw/features/"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def inference(input_path, output_path):
    image = (
        preprocess(Image.open(input_path))
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        image_features = model.encode_image(image)
        with open(output_path, "wb") as f:
            np.save(f, image_features.cpu().numpy())


def process_images(input_folder, output_folder):
    for root, _, files in os.walk(input_folder):
        for filename in tqdm(files):
            input_path = os.path.join(root, filename)
            output_path = os.path.join(
                output_folder,
                os.path.relpath(input_path, input_folder),
            ).replace(".jpg", ".npy")

            os.makedirs(
                os.path.dirname(output_path), exist_ok=True
            )

            inference(input_path, output_path)


if __name__ == "__main__":
    process_images(input_folder, output_folder)
