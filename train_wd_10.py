#!/home/sunhanbo/anaconda3/bin/python
#-*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
from tensorboardX import SummaryWriter
global tensorboard_writer

MOMENTUM = 0.9
WEIGHT_DECAY = 0.001
MILESTONES = [50, 75]
GAMMA = 0.1
EPOCHS = 100

TRAIN_PARAMETER = '''\
# TRAIN_PARAMETER
## loss
CrossEntropyLoss
## optimizer
SGD: base_lr %f momentum %f weight_decay %f
## lr_policy
MultiStepLR: milestones [%s] gamma %f epochs %d
'''%(
0,
MOMENTUM,
WEIGHT_DECAY,
', '.join(str(e) for e in MILESTONES),
GAMMA,
EPOCHS,
)

def train_net(net, train_loader, test_loader, lr, device, prefix):
    global tensorboard_writer
    tensorboard_writer = SummaryWriter(comment = prefix)
    # set net on gpu
    net.to(device)
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr = lr, momentum = MOMENTUM, weight_decay = WEIGHT_DECAY)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones = MILESTONES, gamma = GAMMA)
    # initial test
    eval_net(net, test_loader, 0, device)
    # epochs
    for epoch in range(EPOCHS):
        # train
        net.train()
        scheduler.step()
        for i, (images, labels) in enumerate(train_loader):
            net.zero_grad()
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f'epoch {epoch+1:3d}, {i:3d}|{len(train_loader):3d}, loss: {loss.item():2.4f}', end = '\r')
            tensorboard_writer.add_scalars('train_loss', {'train_loss': loss.item()}, epoch * len(train_loader) + i)
        eval_net(net, test_loader, epoch + 1, device)
        torch.save(net.state_dict(), f'zoo/{prefix}_params.pth')

def eval_net(net, test_loader, epoch, device):
    # set net on gpu
    net.to(device)
    net.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            test_total += labels.size(0)
            outputs = net(images)
            # predicted
            labels = labels.to(device)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == labels).sum().item()
    print('%s After epoch %d, accuracy is %2.4f' % \
          (time.asctime(time.localtime(time.time())), epoch, test_correct / test_total))
    tensorboard_writer.add_scalars('test_acc', {'test_acc': test_correct / test_total}, epoch)

if __name__ == '__main__':
    print(TRAIN_PARAMETER)
