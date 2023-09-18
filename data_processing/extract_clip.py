from sklearn import preprocessing
import torch
import clip
from PIL import Image
import numpy as np

import os
from PIL import Image
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

input_folder = "./raw/frames/"

output_folder = "./raw/features-lation/"

device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)

model = CLIPModel.from_pretrained(
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
)
processor = CLIPProcessor.from_pretrained(
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
)
model = model.to(device)

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def inference(input_path: str, output_path: str):
    with torch.no_grad():
        features = model.get_image_features(
            **processor(
                images=Image.open(input_path),
                return_tensors="pt",
            ).to(device)
        )
        features = features.cpu().detach().numpy()

        with open(output_path, "wb") as f:
            np.save(f, features)


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
