#!/home/sunhanbo/anaconda3/bin/python
#-*-coding:utf-8-*-
import torch
from importlib import import_module
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-g', '--gpu', help = 'select gpu')
parser.add_argument('-d', '--dataset', help = 'select dataset')
parser.add_argument('--nad', help = 'select net attack define')
parser.add_argument('--naw', help = 'select net attack weights')
parser.add_argument('--ndd', help = 'select net defense define')
parser.add_argument('--ndw', help = 'select net defense weights')
parser.add_argument('-p', '--prefix', help = 'select prefix')
args = parser.parse_args()
assert args.gpu
assert args.dataset
assert args.nad
assert args.naw
assert args.ndd
assert args.ndw
assert args.prefix

# dataloader
dataset_module = import_module(args.dataset)
_, test_loader = dataset_module.get_dataloader()
# load net_attack and net defense
net_attack_module = import_module(args.nad)
net_attack = net_attack_module.get_net()
net_attack.load_state_dict(torch.load(args.naw))

net_defense_module = import_module(args.ndd)
net_defense = net_defense_module.get_net()
net_defense.load_state_dict(torch.load(args.ndw))
# attack
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
import FGSM
epsilon_list = [i / 300. for i in range(30 + 1)]
FGSM.eval_attack(net_attack, net_defense, test_loader, epsilon_list, device, args.prefix)
