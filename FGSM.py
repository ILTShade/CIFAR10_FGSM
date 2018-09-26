#!/home/sunhanbo/anaconda3/bin/python
#-*-coding:utf-8-*-
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

def eval_attack(net_attack, net_defense, test_loader, epsilon_list, device, prefix):
    print(prefix)
    tensorboard_writer = SummaryWriter(comment = prefix)
    # set net_attack and net_defense to device
    net_attack.to(device)
    net_defense.to(device)
    # set net_attack and net_defense to eval mode
    net_attack.eval()
    net_defense.eval()
    # loss for gradient cal
    criterion = nn.CrossEntropyLoss()
    # eval attack for all epsilon in epsilon_list
    for index, epsilon in enumerate(epsilon_list):
        adv_correct = 0
        adv_total = 0
        for images, labels in test_loader:
            assert images.size(0) == labels.size(0)
            images.requires_grad_()
            labels = labels.to(device)
            # cal gradient
            outputs = net_attack(images.to(device))
            if isinstance(outputs, tuple):
                outputs = outputs[1]
            loss = criterion(outputs, labels)
            loss.backward()
            # generate adv and pre-treatment
            adv_images = images + epsilon * torch.sign(images.grad)
            adv_images = torch.clamp(adv_images, 0, 1)
            adv_images = torch.round(adv_images * 256.) / 256.
            # generate adv_outputs
            adv_outputs = net_defense(adv_images.to(device))
            if isinstance(adv_outputs, tuple):
                adv_outputs = adv_outputs[1]
            # acc
            adv_correct += torch.sum(torch.max(adv_outputs, 1)[1] == labels).item()
            adv_total += labels.size(0)
        print(epsilon, adv_correct / adv_total)
        tensorboard_writer.add_scalars('attack', {'test_acc': adv_correct / adv_total}, index)
