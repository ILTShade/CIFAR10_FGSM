#!/home/sunhanbo/anaconda3/bin/python
#-*-coding:utf-8-*-
import os
net_info = {'resnet32': {'define': 'resnet32', 'weights': 'zoo/cifar10_resnet32_params.pth'},
            'lenet': {'define': 'lenet', 'weights': 'zoo/cifar10_lenet_params.pth'},
            'lenet_cent': {'define': 'lenet_cent', 'weights': 'zoo/cifar10_lenet_cent_params.pth'},
            }

for key_attack, value_attack in net_info.items():
    for key_defense, value_defense in net_info.items():
        cmd = 'python synthesize_attack.py -g %d -d %s --nad %s --naw %s --ndd %s --ndw %s -p %s'\
              % (2,\
                 'cifar10',\
                 value_attack['define'],
                 value_attack['weights'],
                 value_defense['define'],
                 value_defense['weights'],
                 '%s_attack_%s' % (key_attack, key_defense))
        print(cmd)
        os.system(cmd)
