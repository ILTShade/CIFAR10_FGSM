#!/home/sunhanbo/anaconda3/bin/python
#-*-coding:utf-8-*-
import torch
from importlib import import_module
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', help = 'select gpu')
parser.add_argument('-d', '--dataset', help = 'select dataset')
parser.add_argument('-n', '--net', help = 'select net')
parser.add_argument('-t', '--train', help = 'select train')
parser.add_argument('-p', '--prefix', help = 'select prefix')
args = parser.parse_args()
assert args.gpu
assert args.dataset
assert args.net
assert args.train
assert args.prefix

# dataloader
dataset_module = import_module(args.dataset)
train_loader, test_loader = dataset_module.get_dataloader()
# net
net_module = import_module(args.net)
net = net_module.get_net()
# train
train_module = import_module(args.train)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_module.train_net(net, train_loader, test_loader, device, args.prefix)

