# Imports here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import torchvision
from collections import OrderedDict
import argparse

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(30), # Random rotation by a angle
        transforms.RandomResizedCrop(224), # Crop image to random size and aspect ratio
        transforms.RandomHorizontalFlip(), # Horizontally flip image randomly
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])
}

data_types = ['train', 'valid', 'test']

image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
}


train_loader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
validation_loader = torch.utils.data.DataLoader(image_datasets['valid'], batch_size =64,shuffle = True)
test_loader = torch.utils.data.DataLoader(image_datasets['test'], batch_size = 64, shuffle = True)




dataiter = iter(train_loader)
inputs, labels = dataiter.next()


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
parser.add_argument('--data_path', type=str, default="./flowers/", help='data path')
parser.add_argument('--arch', type=str, default='vgg16', help='Determines which architecture you choose to utilize')
parser.add_argument('--device', default='gpu', type=str, help='run model on cpu or gpu')
parser.add_argument('--epochs', type=int, default=5, help='number of epochs')    

inputs = parser.parse_args()

data_path = inputs.data_dir
arch = inputs.archhidden_layer = cl_inputs.hidden_layer
device = inputs.device
epochs = inputs.epochs
dropout = inputs.dropout
device = torch.device('cuda' if torch.cuda.is_available() and in inputs.gpu else 'cpu')






 
if arch == vgg:# TODO: Build and train your network
    model = models.vgg13(pretrained=True)
elif arch == 'Densenet':
    model = models.densenet121(pretrained=True)
        
# don't compute gradients
for param in model.parameters():
    param.requires_grad = False
    
# create new classifier
classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(25088, 12544)),
    ('relu1', nn.ReLU()),
    ('dropout1', nn.Dropout(p=0.5)),
    ('fc2', nn.Linear(12544, 6272)),
    ('relu2', nn.ReLU()),
    ('dropout2', nn.Dropout(p=0.5)),
    ('fc4', nn.Linear(6272, 102)),
    ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier

criterion = nn.NLLLoss()

optimizer = optim.Adam(model.classifier.parameters(), lr=0.0008)

model.to(device)



epochs = 5
steps = 0
running_loss = 0

training_losses, validation_losses = [], []

for e in range(epochs):
    for inputs, targets in train_loader:
        steps += 1
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    else:
        valid_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)                               
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                valid_loss += batch_loss.item()
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
                  "Validation Loss: {:.3f}.. ".format(valid_loss/len(validation_loader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(validation_loader)))
            
    model.train()


model.eval()

accuracy = 0
test_loss = 0
for ii, (inputs, labels) in enumerate(test_loader):

    inputs, labels = inputs.to(device), labels.to(device)
    logps = model.forward(inputs)
    test_loss += criterion(logps, labels).data[0]


    ps = torch.exp(logps).data

    equality = (labels.data == ps.max(1)[1])

    accuracy += equality.type_as(torch.FloatTensor()).mean()

print("Test Loss: {:.3f}.. ".format(test_loss/len(test_loader)),
      "Test Accuracy: {:.3f}".format(accuracy/len(test_loader)))        
        
model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {'epochs': epochs,
              'training_losses': training_losses,
              'validation_losses': validation_losses,
              'class_to_idx': model.class_to_idx,
              'layers': [25088, 12544, 6272, 102],
              'optimizer_state_dict': optimizer.state_dict(), 
              'model_state_dict': model.state_dict()}

torch.save(checkpoint, 'model_checkpoint.pth')

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pic = Image.open(image)
   
    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_np = image_transform(pic)
    
    return img_np

def predict(image_path, model, topk=topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = process_image(image_path)
    image = image.float().unsqueeze_(0)
    image=image.to(device)
    out = model.forward(image)
        
    pred = F.softmax(out.data, dim = 1)
    
    prob, indices = pred.topk(topk)

    prob = np.array(prob)[0]
    indices = np.array(indices)[0]
    
    idx_to_class = {v:k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[x] for x in indices]
    return(prob,classes)

