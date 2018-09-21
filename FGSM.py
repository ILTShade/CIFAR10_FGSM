#!/home/sunhanbo/anaconda3/bin/python
#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as Transforms
import numpy as np
import mnist
import resnet
from tensorboardX import SummaryWriter
tensorboard_writer = SummaryWriter()

# load model and weights
train_loader, test_loader = mnist.get_data()
net = resnet.get_resnet()
net.load_state_dict(torch.load('zoo/resnet_100_params.pth'))

# transfer adv_images
def transfer_adv_images(adv_images):
    # clap and interger
    adv_images = torch.round(torch.clamp(adv_images, 0, 1) * 256) / 256
    # deep copy
    adv_images = torch.tensor(data = adv_images.data, dtype = adv_images.dtype, device = device, requires_grad = True)
    return adv_images

# fgsm
# net, input_images, input_labels must inthe same device
def FGSM(net, input_images, input_labels, criterion):
    net.zero_grad()
    outputs = net(input_images)
    loss = criterion(outputs, input_labels)
    loss.backward()
    adv_images = input_images + epsilon * torch.sign(labels.size(0) * input_images.grad)
    adv_images = transfer_adv_images(adv_images)
    return adv_images

# set device and criterion
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
img_transform = Transforms.ToPILImage()

# grad for every epoch
net.eval()
net.to(device)
recode = []
for e in range(30 + 1):
    epsilon = e / 100
    adv_correct = 0
    save_count = 0
    for images, labels in test_loader:
        # check images and labels size, transfer them to appointed device
        assert images.size(0) == labels.size(0)
        input_images = torch.tensor(data = images.data, dtype = images.dtype, device = device, requires_grad = True)
        input_labels = torch.tensor(data = labels.data, dtype = labels.dtype, device = device, requires_grad = False)
        # generate adv images and get its acc
        adv_images = FGSM(net, input_images, input_labels, criterion)
        # generate predict for input and adv input
        _, input_predicts = torch.max(net(input_images), 1)
        _, adv_predicts = torch.max(net(adv_images), 1)
        adv_correct += (adv_predicts == input_labels).sum().item()
        # save image
        save_index = torch.nonzero((input_predicts == input_labels) & (adv_predicts != input_labels))
        for i in save_index:
            if save_count > 0:
                break
            i = i.item()
            # origin image
            ori_image = input_images[i].cpu().detach()
            ori_image = np.array(img_transform(ori_image))
            # adv image
            adv_image = adv_images[i].cpu().detach()
            adv_image = np.array(img_transform(adv_image))
            # perturbation
            per_image = np.abs(adv_image.astype(np.float) - ori_image.astype(np.float)).astype(np.uint8)
            # save
            tensorboard_writer.add_image(f'SAVE_{save_count:06d}', ori_image, 1)
            tensorboard_writer.add_image(f'SAVE_{save_count:06d}', adv_image, 2)
            tensorboard_writer.add_image(f'SAVE_{save_count:06d}', per_image, 3)
            save_count = save_count + 1
    print(adv_correct)
    recode.append([epsilon, adv_correct])
print(recode)
