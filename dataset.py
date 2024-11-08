import cv2
import torch
from torchvision import transforms
from torch.utils import data
from math import log2
import os
from config import CHANNELS_IMG, BATCH_SIZES
from config import ROOT


class Dataset(data.Dataset):
    def __init__(self, root):
        super(Dataset, self).__init__()
        self.root = root
        self.img_names = os.listdir(root)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(CHANNELS_IMG)],
                [0.5 for _ in range(CHANNELS_IMG)],
            ),
        ])

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.img_names[index])
        img = cv2.imread(img_path)
        img = self.transform(img)
        label = torch.tensor([1])
        return img, label

    def __len__(self):
        return len(self.img_names)


def getloader(root, image_size):
    dataset = Dataset(root)
    batch_size = BATCH_SIZES[int(log2(image_size / 4))]
    loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    return dataset,loader


if __name__ == '__main__':
    image_size = 256
    dataset, loader = getloader(ROOT,  image_size)

    imgs, labels = next(iter(loader))
    print(imgs.shape)
    print(labels.shape)
