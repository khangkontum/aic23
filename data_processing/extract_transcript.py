from multiprocessing import Pool
from subprocess import call
from tqdm import tqdm
import json
import os
import torch
import whisper

import torch.multiprocessing as mp

input_folder = "./raw/sounds"
output_folder = "./raw/transcript"

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base", device)

def inference_and_save(model, input_path, output_path):
    # input_path, output_path = paths
    result = model.transcribe(input_path)

    with open(output_path, "w+") as f:
        json.dump(result, f, indent = 6, ensure_ascii=False)


if __name__ == "__main__":

    paths = []

    processes = []
    counter = 0
    for root, _, files in os.walk(input_folder):
        for filename in tqdm(files):
            input_path = os.path.join(root, filename)
            output_path = os.path.join(
                output_folder,
                filename,
            ).replace(".mp3", ".json")

            paths.append((input_path, output_path))

            p = mp.Process(target=inference_and_save, args=(model,input_path,output_path))
            p.start()
            processes.append(p)

            counter += 1
            if counter % 4 == 0:
                for p in processes:
                    p.join()

            del processes[:]
