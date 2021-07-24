from PIL import Image
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from source.classes.herbarium import Herbarium
from source.functions import load_coco_data
from source.functions import create_data_loader

def main():
    
    train_dir = '/Volumes/CF_Lacie_P2/train/'
    test_dir = '/Volumes/CF_Lacie_P2/test/'
    meta_filename = 'metadata.json'
    herb = Herbarium(train_dir, test_dir, meta_filename)
    #herb.load_train_data()
    #herb.test()
    #herb.step_train()

    coco_data, nb_classes = load_coco_data(train_dir, meta_filename)[0:98]

    herb.train_data_getter, _= create_data_loader(train_dir, coco_data, 'file_name', 'category_id', herb.transform, herb.batch)
    herb.nb_classes = nb_classes
    herb.init_model()

if __name__ == "__main__":
    main()