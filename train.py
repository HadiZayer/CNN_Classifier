import torch
import numpy as np
from torch.autograd import Variable
from torch import optim
from torch import nn
import torchvision
import os
import imageio
import argparse
from models import *
from dataloaders import *

parser = argparse.ArgumentParser(description='train a cnn')
parser.add_argument('--epochs', type=int, help='number of training epochs', default=1)
parser.add_argument('--batch_size', type=int, help='training batch size', default=64)
parser.add_argument('--model', type=str, help='choose model: \'basic_cnn\' or \'vgg\'', default='basic_cnn')
parser.add_argument('--input_channels', type=int, help='number of input channels', default=1)
parser.add_argument('--output_classes', type=int, help='number of output classes', default=10)
parser.add_argument('--load_model', type=str, help='path to model to load (if applicable)')
parser.add_argument('--data_path', type=str, required=True, help='path of training data')
parser.add_argument('--saved_model_path', type=str, help='path to save trained model', default='model.pt')
parser.add_argument('--learning_rate', type=float, help='training learning rate', default=1e-3)
parser.add_argument('--validation', type=float, help='percentage of training data used for validation', default=0.2)
parser.add_argument('--gpu', type=bool , help='use GPU', default=False)

args = parser.parse_args()

def train(model, train_x, train_y, learning_rate=1e-3, batch_size=64, epochs=1, gpu=False):
    model.train()
    if(gpu):
        model.cuda()
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        
    dataset = torch.utils.data.TensorDataset(train_x, train_y)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=batch_size)
    
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.1)
    cross_entropy = nn.CrossEntropyLoss()
    for i in range(epochs):
        for train_x, train_y in dataloader:
            train_x = train_x.unsqueeze(1) #num of channels is 1
            train_x = Variable(train_x).float()
            train_y = Variable(train_y)
            output = model(train_x)

            optimizer.zero_grad()
            loss = cross_entropy(output, train_y)
            loss.backward()
            optimizer.step()
        print('finished epoch ' + str(i))

def test(model, test_x, test_y, batch_size=64):
    model.eval()
    counter = 0.0
    
    dataset = torch.utils.data.TensorDataset(test_x, test_y)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=batch_size)
    for x, y in dataloader:
        x = x.unsqueeze(1) #num of channels is 1
        x = Variable(x).float()
        output = model(x)
        _, indices = output.max(1)
        counter += sum(indices.data.numpy() == y)
    return counter/len(test_y)

if __name__ == '__main__':
	assert args.model.lower() == 'basic_cnn' or 'vgg'
	model = None
	if args.model.lower() == 'basic_cnn':
		model = basic_cnn(args.input_channels, args.output_classes)
	elif args.model.lower() == 'vgg':
		model = vgg(args.input_channels, args.output_classes)

	if args.load_model != None:
		model.load_state_dict(torch.load(args.load_model))

	train_x, train_y, test_x, test_y = load_training_data(args.data_path, args.output_classes, args.validation)
	train(model, train_x, train_y, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, gpu=args.gpu)
	accuracy = test(model, test_x, test_y)
	print('accuracy with validation is ' + str(accuracy))

	torch.save(model.state_dict(), args.saved_model_path)


