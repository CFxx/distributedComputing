from typing import ForwardRef
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
        self.iter_data = None
        self.test_data_getter = None
        self.nb_classes = 0
        self.epochs = 1
        self.lr = 0.01
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.loss = None
        self.batch = 16
        self.image_size = 32
        self.transform = transforms.Compose( [transforms.ToTensor(), 
                                        transforms.Resize((self.image_size, self.image_size)), 
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    
    def load_train_data(self, limit=1024):

        coco_data, self.nb_classes = load_coco_data(self.train_dir, self.meta_filename)
        print(f'Total classes : {self.nb_classes}')
        self.train_data_getter, _= create_data_loader(self.train_dir, coco_data, 'file_name', 'category_id', self.transform, self.batch, limit)
        print(f'Total samples in loader :{len(self.train_data_getter)}')
        self.init_model()

    def init_model(self):

        self.model = torchvision.models.resnet34()
        self.model.fc = nn.Linear(512, self.nb_classes, bias=True)
        self.model = self.model.to(self.device)

         # loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def forward_step(self):
        
        if self.iter_data == None:
            self.iter_data = iter(self.train_data_getter)
        
        self.model = self.model.train()
        x, y = next(self.iter_data)
        y = y.to(self.device)
        x = x.to(self.device)
        return self.model(x), y
    
    def backward_step(self, y_pred, y, optimized=False):

        self.loss = self.criterion(y_pred, y)
        self.optimizer.zero_grad()
        self.loss.backward() # gradient calculation
        if optimized:
            self.optimizer.step()
            return self.loss.detach().item()
        else:
            return None

    def train(self, epochs = 3):

        print("start training")
        # forward pass
        for epoch in range(epochs):
            print(f'epoch {epoch}')
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
            print(f'value i : {i}')
            print(f'Epoch : {epoch} | Loss : {(progress_loss/i):.4}')
    
    def step_train(self, epochs=1, verbose=False):

        for i in range(epochs):
            progress_loss = 0.0
            iterate = True
            #count = 0
            while iterate:
                try:
                    y_pred, y = self.forward_step()
                    progress_loss += self.backward_step(y_pred, y, True)
                    #if count > 0:
                    #    self.model.load_state_dict(weights)
                    #    print("resetting weights")
                    #else:
                    #    weights = self.model.state_dict()

                    #count += 1
                    #print(f'Epoch : {i+1} | Loss : {(progress_loss/count):.4}')
                except StopIteration:
                    self.model.eval()
                    self.iter_data = None
                    iterate = False
            if verbose:
                print(f'Epoch : {i+1} | Loss : {(progress_loss/64):.4}')
    
    def test(self):
        iterate = True
        while iterate:
            try:
                y_pred, y = self.forward_step()
                #print("parameters before changing : ")
                #print(self.model.parameters())
                self.backward_step(y_pred, y, True)
                
                #print("parameters after changing : ")
                #print(self.model.parameters)

                print(self.model.state_dict())

                input()
            except StopIteration:
                iterate = False