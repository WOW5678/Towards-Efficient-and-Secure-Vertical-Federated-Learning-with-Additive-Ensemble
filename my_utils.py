#! usr/bin/env python 39
# -*- coding: utf-8 -*- 
# @Time  : 2023/3/29 14:46
# @Author: sharonswang   
# @File  : my_utils.py

import matplotlib.pyplot as plt
import torch
import torch.functional as F

def plot_mnist(images,labels):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        print(images[i])
        plt.imshow(images[i], cmap='gray', interpolation='none')
        plt.title('Ground Truth:{}'.format(labels[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


def frezze_network(passive_model_a, passive_model_b, active_model_c, flag='trainC'):
    if flag == 'trainC':
        # 需要将passive_model_a和passive_model_b模型的参数进行固定, 设置active_model_c模型的参数可训练
        #print(passive_model_a.named_parameters)
        for param in passive_model_a.parameters():
            param.requires_grad = False
        for param in passive_model_b.parameters():
            param.requires_grad = False
        for param in active_model_c.parameters():
            param.requires_grad = True
    elif flag == 'trainA':
        for param in passive_model_a.parameters():
            param.requires_grad = True
        for param in passive_model_b.parameters():
            param.requires_grad = False
        for param in active_model_c.parameters():
            param.requires_grad = False
    elif flag == 'trainB':
        for param in passive_model_a.parameters():
            param.requires_grad = False
        for param in passive_model_b.parameters():
            param.requires_grad = True
        for param in active_model_c.parameters():
            param.requires_grad = False


def int2onehot(config, labels, num_classes):
    labels = labels.view(-1, 1)
    onehot = torch.zeros(labels.shape[0], num_classes).to(config.device)
    onehot.scatter_(1, labels, 1)

    return onehot



