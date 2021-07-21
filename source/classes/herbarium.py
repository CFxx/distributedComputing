import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from ..functions import load_coco_data
from ..functions import create_data_loader
from .sampleGetter import SampleGetter

class Herbarium:

    def __init__(self, train_dir, test_dir, meta_filename = None):

        self.train_dir = train_dir
        self.test_dir = test_dir
        self.meta_filename = meta_filename

        self.train_data_getter = None
        self.test_data_getter = None
        self.nb_classes = 0
        self.epochs = 2
        self.lr = 0.01
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = torchvision.models.resnet34()
        self.model.fc = nn.Linear(512, self.nb_classes, bias=True)
        self.model = self.model.to(self.device)

        self.batch = 16
        self.image_size = 32
        self.transform = transforms.Compose( [transforms.ToTensor(), 
                                        transforms.Resize((self.image_size, self.image_size)), 
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        # loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
    def load_train_data(self, limit=1000):

        coco_data = load_coco_data(self.train_dir, self.meta_filename)
        self.train_data_getter = create_data_loader(self.train_dir, coco_data, 'file_name', 'category_id', self.transform, self.batch, limit)

    def train(self):

        # forward pass
        for epoch in range(self.epochs):
            progress_loss = 0.0
            
            # model training mode
            self.model = self.model.train()

            for i, (images, labels) in enumerate(self.train_data_getter):
                images = images.to(self.device)
                labels = labels.to(self.device)
                output = self.model(images)

                loss = self.criterion(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                progress_loss += loss.detach().item()

            self.model.eval()
            print(f'Epoch : {epoch} | Loss : {(progress_loss/i):.4}')