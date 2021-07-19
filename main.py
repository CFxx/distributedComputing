from PIL import Image
import os
import numpy as np
import pandas as pd
import json
import torchvision
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from source.classes.herbarium import Herbarium


def main():
    
    train_dir = '/Volumes/CF_Lacie_P2/train/'
    test_dir = '/Volumes/CF_Lacie_P2/test/'
    meta_filename = 'metadata.json'
    herb = Herbarium(train_dir, test_dir, meta_filename)
    herb.load()
    herb.train()
    #X, Y = next(iter(herb.train_data_getter))
    #print(X.shape, Y.shape)




if __name__ == "__main__":
    main()