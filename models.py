#! usr/bin/env python 39
# -*- coding: utf-8 -*- 
# @Time  : 2023/3/30 11:09
# @Author: sharonswang   
# @File  : savedModels.py

import torch
import torch.nn as nn
import numpy as np

# 客户端模型
class BottomModel(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(BottomModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            # 实验证实batchNormalization并不会提升实验效果
            #nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer(x)
        return out

class TopModel(nn.Module):
    def __init__(self, hidden_size, num_classes=2):
        super(TopModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # 实验证实batchNormalization并不会提升实验效果
            #nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            #nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, num_classes)
        )


    def forward(self, x):
        out = self.layer(x)
        return out


class RegressionBottomModel(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(RegressionBottomModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            # 实验证实batchNormalization并不会提升实验效果
            # nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.layer(x)
        return out

class RegressionTopModel(nn.Module):
    def __init__(self, hidden_size, num_classes=1):
        super(RegressionTopModel, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            # 实验证实batchNormalization并不会提升实验效果
            # nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, num_classes)

        )

    def forward(self, x):
        out = self.layer(x)
        return out

class Centralized(nn.Module):
    def __init__(self, in_size, hidden_size, mum_classes=2):
        super(Centralized, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            #nn.Dropout(0.5),
            # 实验证实batchNormalization并不会提升实验效果
            #nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            #nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            #nn.Dropout(0.5),
            # 实验证实batchNormalization并不会提升实验效果
            #nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, mum_classes),
        )


    def forward(self, x):
        out = self.layer(x)
        return out


class Cifar10_BottomModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, hidden_size):
        super(Cifar10_BottomModel, self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(out_channels, 16, kernel_size),
            nn.Flatten(),

        )
        conv_out_size = self._get_conv_out((in_channels, 32, 32)) # 传入图片的size
        print('conv_out_size:', conv_out_size)
        self.layer = nn.Sequential(
            nn.Linear(conv_out_size, hidden_size),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = x.view(-1, self.in_channels, 32, 32)
        x = self.conv1(x)
        x = self.layer(x)
        return x

    def _get_conv_out(self, shape):
        out = self.conv1(torch.zeros(1, *shape))
        return int(np.prod(out.size()))

class Cifar10_TopModel(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(Cifar10_TopModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = self.layers(x)
        return x