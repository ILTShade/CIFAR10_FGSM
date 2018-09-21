#!/home/sunhanbo/anaconda3/bin/python
#-*-coding:utf-8-*-
import resnet
def get_net():
    net = resnet.ResNet(resnet.BasicBlock, [3, 4, 6, 3], num_classes = 10)
    return net
if __name__ == '__main__':
    net = get_net()
    print(net)
    print('这是resnet32网络，要求输入尺寸必为3x32x32，输出仅输出10分类的分类结果')
