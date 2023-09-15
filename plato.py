import pickle
import os
import numpy as np
import clip
import torch
import numpy as np
from tqdm import tqdm
import copy

class Plato:
    def __init__(self, dataset):
        if dataset == None:
            raise exit
        self.dataset = []
        print("Converting dataset")
        for sample in tqdm(dataset):
            dict_sample = {
                'video': sample['video'],
                'filepath': sample['filepath'],
                'mapped_frameid': sample['mapped_frameid'],
                'clip_embedding': sample['clip_embedding'],
                'frameid': sample['frameid']
            }
            self.dataset.append(dict_sample)
            
        print("Concat embedding array into a big array")
        self.stack_vector = []
        npy_dir = f"/home/hoang-chau/Documents/Python/AI-Challenge-2023_Aristotle/data/clip-features-vit-b32/"
        for file in sorted(os.listdir(npy_dir)):
            file_path = os.path.join(npy_dir, file)
            clip_embedding = np.load(file_path)
            self.stack_vector.append(clip_embedding)
        self.stack_vector = np.vstack(self.stack_vector)
        self.model = self._load_model()

    def _load_model(self):
        # option 16 32-not suitable
        model, _ = clip.load("ViT-B/32", device="cpu")
        model.eval()
        return model

    def featurize_text(self, text):
        if len(text) > 77:
            text = text[:76]
        text_tokens = clip.tokenize([text]).cpu()
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens).float()
        return text_features

    def predict(self, text_features, top=1000):
        text_features = self.featurize_text(text_features)
        vector_dataset = self.stack_vector @ text_features.numpy().T
        _, stacked_dataset = zip(
            *sorted(zip(vector_dataset.squeeze(), self.dataset), reverse=True, key=lambda x: x[0]))
        return copy.deepcopy(
            np.asarray(stacked_dataset)[:top]
        )

    def batch_ranking(self, queries):
        result = []
        for i, query in enumerate(queries):
            print(f'Processing query {i+1} ...')
            result.append(self.predict(query))
        return result

    def predict_csv(self, query_dir):
        queries = []
        with open(query_dir, "r") as f:
            for line in f:
                queries.append(line.strip())
        return self.batch_ranking(queries)

    def save(self, path):
        pickle.dump(self, open(path, 'wb'))

    @staticmethod
    def load(path):
        return pickle.load(open(path, 'rb'))

