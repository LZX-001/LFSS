
import numpy as np


from torchvision.datasets import ImageFolder
class CustomImagenet(ImageFolder):
    def __init__(self, **kwds):
        super().__init__(**kwds)
        targets = np.asarray([s[1] for s in self.samples])
        self.targets = targets
        self.img_num = len(self.samples)
        print(self.img_num)
        print(np.unique(self.targets))

    def __getitem__(self, idx):
        path,target = self.samples[idx]
        img = self.loader(path)
        # img = Image.fromarray(img).convert('RGB')
        imgs = self.transform(img)
        # target = self.targets[idx]
        return imgs,target,idx

