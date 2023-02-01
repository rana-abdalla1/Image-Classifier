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


parser = argparse.ArgumentParser()

parser.add_argument('--json', type=str, default='cat_to_name.json')
parser.add_argument('--checkpoint', type=str, default='model_checkpoint.pth')
parser.add_argument('--topk', type=int, default=5)
parser.add_argument('--device', default='gpu', type=str)
args = parser.parse_args()

if args.checkpoint:
    checkpoint = args.checkpoint

if args.topk:
    topk = args.topk
if args.json:
    filepath = args.json
if args.gpu:
    torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



training_loader, testing_loader, validation_loader = train.load_data()


train.load_checkpoint(path)


with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)


probabilities = train.predict(path_image, model)


labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
probability = np.array(probabilities[0][0])


i=0
while i < number_of_outputs:
    print("{} with a probability of {}".format(labels[i], probability[i]))
    i += 1