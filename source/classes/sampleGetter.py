import os
from torch.utils.data import Dataset
from PIL import Image

class SampleGetter(Dataset):

    def __init__(self, dir, fnames, labels, transform):

        self.dir = dir
        self.fnames = fnames
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.fnames)
        
    def __getitem__(self, index):

        loaded_item = Image.open(os.path.join(self.dir, self.fnames[index]))
        return self.transform(loaded_item), self.labels[index]