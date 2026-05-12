import torch
import torch.utils.data
import torchvision

DATA_DIR = ""
class TrainSet(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.transform = transform

        self.imgs = torchvision.datasets.ImageFolder(root=DATA_DIR, transform=self.transform)

    def __getitem__(self, idx):
        sample = {'img': self.imgs[idx][0]}
        return sample

    def __len__(self):
        return len(self.imgs)
