import whisper
from tqdm import tqdm
import torch
import os
import json

input_folder = "./raw/sounds"
output_folder = "./raw/transcript"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("large", device)

if __name__ == "__main__":
    for root, _, files in os.walk(input_folder):
        for filename in tqdm(files):
            input_path = os.path.join(root, filename)
            output_path = os.path.join(
                output_folder,
                filename,
            ).replace(".mp3", ".json")

            result = model.transcribe(input_path)

            with open(output_path, "w+") as f:
                json.dump(result, f, indent = 6, ensure_ascii=False)
