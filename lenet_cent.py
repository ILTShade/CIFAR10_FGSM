#!/home/sunhanbo/anaconda3/bin/python
#-*-coding:utf-8-*-
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 32, 5, stride = 1, padding = 2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 5, stride = 1, padding = 2)
        self.prelu1_2 = nn.PReLU()
        
        self.conv2_1 = nn.Conv2d(32, 64, 5, stride = 1, padding = 2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 5, stride = 1, padding = 2)
        self.prelu2_2 = nn.PReLU()
        
        self.conv3_1 = nn.Conv2d(64, 128, 5, stride = 1, padding = 2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 5, stride = 1, padding = 2)
        self.prelu3_2 = nn.PReLU()
        
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.prelu_fc1 = nn.PReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)
        
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)
        
        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)
        
        x = x.view(x.size(0), 128 * 4 * 4)
        x = self.prelu_fc1(self.fc1(x))
        y = self.fc2(x)

        return x, y

def get_net():
    net = LeNet(10)
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            module.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(module, nn.BatchNorm2d):
            module.weight.data.fill_(1)
            module.bias.data.zero_()
    return net

if __name__ == '__main__':
    net = get_net()
    print(net)
    print('这是lenet网络，要求输入尺寸必为3x32x32，输出为(128维特征，10维分类结果)')
