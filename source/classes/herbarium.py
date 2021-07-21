import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from ..functions import load_coco_data
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
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    def load(self):

        train_limit = 1000
        image_size = 32
        nb_batch = 16
        transform = transforms.Compose( [transforms.ToTensor(), 
                                        transforms.Resize((image_size, image_size)), 
                                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        train_coco_data = load_coco_data(self.train_dir, self.meta_filename, 'id', ['image_id'])
        self.nb_classes = len(train_coco_data['category_id'].value_counts())
        train_coco_data = train_coco_data[0:train_limit]
        X_Train, Y_Train = train_coco_data['file_name'].values, train_coco_data['category_id'].values
        self.train_data_getter = DataLoader(SampleGetter(self.train_dir, X_Train, Y_Train, transform), batch_size=nb_batch, shuffle=True)
    
    def train(self):
        
        model = torchvision.models.resnet34()
        model.fc = nn.Linear(512, self.nb_classes, bias=True)
        model = model.to(self.device)

        # loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # forward pass
        for epoch in range(self.epochs):
            progress_loss = 0.0
            
            # model training mode
            model = model.train()

            for i, (images, labels) in enumerate(self.train_data_getter):
                images = images.to(self.device)
                labels = labels.to(self.device)
                output = model(images)
                #print("output : ", output)
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                progress_loss += loss.detach().item()

            model.eval()
            print(f'Epoch : {epoch} | Loss : {(progress_loss/i):.4}')