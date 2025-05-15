import torch
from torchvision.datasets import CIFAR100
from PIL import Image
import numpy as np


class CustomCIFAR100(CIFAR100):
    def __init__(self, root, train, transform=None, download=True):
        super().__init__(root, train, transform, download)
        #
        train_dataset = CIFAR100(root=root, train=True)
        test_dataset = CIFAR100(root=root, train=False)

        self.data = np.concatenate([train_dataset.data, test_dataset.data], axis=0)
        self.targets = np.concatenate([train_dataset.targets, test_dataset.targets], axis=0)
        self.targets = superclass(self.targets)
        class_num = len(np.unique(self.targets))
        self.idxsPerClass = [np.where(np.array(self.targets) == idx)[0] for idx in range(class_num)]
        self.idxsNumPerClass = [len(idxs) for idxs in self.idxsPerClass]
        print(self.idxsNumPerClass)

    def __getitem__(self, idx):
        img = self.data[idx]
        img = Image.fromarray(img).convert('RGB')
        imgs = self.transform(img)
        target = self.targets[idx]
        return imgs, target, idx
def superclass(targets):
    targets = np.array(targets)
    super_classes = [
        [72, 4, 95, 30, 55],
        [73, 32, 67, 91, 1],
        [92, 70, 82, 54, 62],
        [16, 61, 9, 10, 28],
        [51, 0, 53, 57, 83],
        [40, 39, 22, 87, 86],
        [20, 25, 94, 84, 5],
        [14, 24, 6, 7, 18],
        [43, 97, 42, 3, 88],
        [37, 17, 76, 12, 68],
        [49, 33, 71, 23, 60],
        [15, 21, 19, 31, 38],
        [75, 63, 66, 64, 34],
        [77, 26, 45, 99, 79],
        [11, 2, 35, 46, 98],
        [29, 93, 27, 78, 44],
        [65, 50, 74, 36, 80],
        [56, 52, 47, 59, 96],
        [8, 58, 90, 13, 48],
        [81, 69, 41, 89, 85],
    ]
    import copy
    copy_targets = copy.deepcopy(targets)
    for i in range(len(super_classes)):
        for j in super_classes[i]:
            targets[copy_targets == j] = i
    return targets