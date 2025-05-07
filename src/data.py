from torch.utils.data import Dataset
import numpy as np
import einops
import csv
import torch


SAMPLES = 500
IN_DIM = 30
OUT_DIM = 20


class GenericDataset(Dataset):
    def __init__(self):
        rng = np.random.default_rng()
        self.x = rng.normal(size=(SAMPLES, IN_DIM)).astype(np.float32)
        self.y = 500 * rng.normal(size=(SAMPLES, OUT_DIM)).astype(np.float32)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

    def __len__(self):
        return len(self.x)

    def get_in_out_size(self):
        return self.x.shape[1], self.y.shape[1]


class FashionDataset(Dataset):
    def __init__(self, path: str):
        self.path = path
        self.x, self.y = self.load()

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

    def __len__(self):
        return len(self.x)

    def load(self):
        # opening the CSV file
        with open(self.path, mode="r") as file:
            images = list()
            classes = list()
            # reading the CSV file
            csvFile = csv.reader(file)
            # displaying the contents of the CSV file
            header = next(csvFile)
            limit = 1000
            for line in csvFile:
                if limit < 1:
                    break
                classes.append(int(line[:1][0]))
                images.append([int(x) for x in line[1:]])
                limit -= 1
            classes = torch.tensor(classes, dtype=torch.long)
            images = torch.tensor(images, dtype=torch.float32)
            images = einops.rearrange(images, "n (w h) -> n w h", w=28, h=28)
            images = einops.repeat(
                images, "n w h -> n c (w r_w) (h r_h)", c=1, r_w=8, r_h=8
            )
            return (images, classes)



if __name__ == "__main__":
    path = "fashion-mnist_train.csv"
    dataset = FashionDataset(path=path)
    print(f"len: {len(dataset)}")
    print(f"first shape: {dataset[0][0].shape}")
    mean = einops.reduce(dataset[:10], "n w h -> w h", "mean")
    print(f"mean shape: {mean.shape}")
