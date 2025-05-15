import torch
from torchvision.datasets import CIFAR10
from PIL import Image
import numpy as np




class CustomCIFAR10(CIFAR10):
    def __init__(self, root, train, transform=None, download=True):
        super().__init__(root, train, transform, download)
        #
        train_dataset = CIFAR10(root=root, train=True)
        test_dataset = CIFAR10(root=root, train=False)

        self.data = np.concatenate([train_dataset.data, test_dataset.data], axis=0)
        self.targets = np.concatenate([train_dataset.targets, test_dataset.targets], axis=0)
        self.idxsPerClass = [np.where(np.array(self.targets) == idx)[0] for idx in range(10)]
        # print(self.idxsPerClass)
        self.idxsNumPerClass = [len(idxs) for idxs in self.idxsPerClass]
        self.data_number = len(self.data)
        print(self.idxsNumPerClass)
    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')
        imgs = self.transform(img)
        target=self.targets[idx]
        return imgs,target,idx


