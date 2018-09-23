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

parser = argparse.ArgumentParser(description='predict classes in test data')
parser.add_argument('--model', type=str, help='choose model: \'basic_cnn\' or \'vgg\'', default='basic_cnn')
parser.add_argument('--input_channels', type=int, help='number of input channels', default=1)
parser.add_argument('--output_classes', type=int, help='number of output classes', default=10)
parser.add_argument('--load_model', type=str, required=True, help='path to model to load (if applicable)')
parser.add_argument('--data_path', type=str, required=True, help='path of training data')
parser.add_argument('--output_path', type=str, help='path to write output file', default='output.csv')
parser.add_argument('--gpu', type=bool, help='use GPU', default=False)

args = parser.parse_args()

def predict(model, test_x, ids, output_file, gpu=False):
    model.eval()
    if(gpu):
        model.cuda()
        test_x = test_x.cuda()

    with open(output_file, 'w') as f:
        for x, i in zip(test_x, ids):
            x = x.unsqueeze(0)
            x = x.unsqueeze(1) #num of channels is 1
            x = Variable(x).float()
            output = model(x)
            _, indices = output.max(1)
            f.write(i + ',' + str(indices.data.numpy()[0]) + '\n')

if __name__ == '__main__':
	assert args.model.lower() == 'basic_cnn' or 'vgg'
	model = None
	if args.model.lower() == 'basic_cnn':
		model = basic_cnn(args.input_channels, args.output_classes)
	elif args.model.lower() == 'vgg':
		model = vgg(args.input_channels, args.output_classes)

	model.load_state_dict(torch.load(args.load_model))
	test_data, ids = load_test_data(args.data_path)
	predict(model, test_data, ids, args.output_path)
