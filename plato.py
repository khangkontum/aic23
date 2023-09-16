import pickle
import numpy as np
import time
import clip
import torch
import numpy as np
from tqdm import tqdm
import copy
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords


def featurize(model, text):
    if len(text) > 77:
        text = text[:76]
    text_tokens = clip.tokenize([text]).cpu()
    with torch.no_grad():
        text_features = model.model.encode_text(
            text_tokens
        ).float()
    return text_features

def get_key_frame(model, video, keyframe):
    for sample in model.dataset:
        if sample["video"] == video and sample["frameid"] == keyframe:
            return sample["mapped_frameid"], sample["youtube_url"]
    
    return "undifined", "undifined"


def predict(model, text, method="cosine", top=1000):
    text_features = featurize(model, text)
    if method == "cosine":
        vector_dataset = []
        for sample in tqdm(model.dataset):
            a = np.array(sample["clip_embedding"])
            vector_dataset.append(
                cosine_similarity(
                    text_features, a.reshape(1, 512)
                )
            )

        print("Sorting...")
        start = time.time()
        _, cosine_dataset = zip(
            *sorted(
                zip(vector_dataset, model.dataset),
                reverse=True,
                key=lambda x: x[0],
            )
        )

        print(
            "Sorting Done in ",
            time.time() - start,
            "seconds",
        )
        return copy.deepcopy(
            np.asarray(cosine_dataset)[:top]
        ),

    elif method == "dot":
        vector_dataset = (
            model.stack_vector @ text_features.numpy().T
        )

        _, stacked_dataset = zip(
            *sorted(
                zip(
                    vector_dataset.squeeze(), model.dataset
                ),
                reverse=True,
                key=lambda x: x[0],
            )
        )
        return copy.deepcopy(
            np.asarray(stacked_dataset)[:top]
        )


class Plato:
    def __init__(self, dataset):
        if dataset == None:
            return
        self.dataset = []
        print("Converting dataset")
        for sample in tqdm(dataset):
            dict_sample = {
                "video": sample["video"],
                "filepath": sample["filepath"],
                "mapped_frameid": sample["mapped_frameid"],
                "clip_embedding": sample["clip_embedding"],
                "frameid": sample["frameid"],
            }
            self.dataset.append(dict_sample)
        self.stack_vector = []
        count = 0
        print("Stacking dataset")
        for sample in tqdm(dataset):
            count += 1
            a = sample["clip_embedding"].reshape(1, 512)
            self.stack_vector.append(a)
        self.stack_vector = np.array(self.stack_vector)
        self.model = self._load_model()

    def _load_model(self):
        # option 16 32-not suitable
        model, _ = clip.load("ViT-B/16", device="cpu")
        model.eval()
        return model

    def featurize_text(self, text):
        if len(text) > 77:
            text = text[:76]
        text_tokens = clip.tokenize([text]).cpu()
        with torch.no_grad():
            text_features = self.model.encode_text(
                text_tokens
            ).float()
        return text_features

    def predict(
        self, text_features, method="cosine", top=1000
    ):
        # text_features = self.featurize_text(text)
        if method == "cosine":
            vector_dataset = []
            for sample in tqdm(self.dataset):
                a = np.array(sample["clip_embedding"])
                vector_dataset.append(
                    cosine_similarity(
                        text_features, a.reshape(1, 512)
                    )
                )
            print("Sorting...")
            start = time.time()
            _, cosine_dataset = zip(
                *sorted(
                    zip(vector_dataset, self.dataset),
                    reverse=True,
                    key=lambda x: x[0],
                )
            )
            print(
                "Sorting Done in ",
                time.time() - start,
                "seconds",
            )
            return np.asarray(cosine_dataset)[:top]

        elif method == "dot":
            vector_dataset = (
                self.stack_vector @ text_features.numpy().T
            )
            _, stacked_dataset = zip(
                *sorted(
                    zip(
                        vector_dataset.squeeze(),
                        self.dataset,
                    ),
                    reverse=True,
                    key=lambda x: x[0],
                )
            )
            return np.asarray(stacked_dataset)[:top]

    def batch_ranking(self, queries, method="cosine"):
        result = []
        i = 0
        for query in queries:
            print(f"Processing query {i+1} ...")
            i += 1
            result.append(self.predict(query, method))
        return result

    def predict_csv(self, query_dir, method="cosine"):
        queries = []
        with open(query_dir, "r") as f:
            for line in f:
                word_list = line.split(" ")
                filtered_words = [
                    word.lower()
                    for word in word_list
                    if word.lower()
                    not in stopwords.words("english")
                ]
                new_line = " ".join(filtered_words)
                queries.append(new_line.strip())
        return self.batch_ranking(queries, method)

    def save(self, path):
        pickle.dump(self, open(path, "wb"))

    @staticmethod
    def load(path):
        return pickle.load(open(path, "rb"))


if __name__ == "__main__":
    model = Plato.load("plato.pkl")
    result = predict(
        model, "boy with orange shirt", "dot", 200
    )
    print(result)
