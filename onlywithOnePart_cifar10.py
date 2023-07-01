#! usr/bin/env python 39
# -*- coding: utf-8 -*-
# @Time  : 2023/3/23 13:26
# @Author: sharonswang
# @File  : main.py

import numpy.ma
import torch
import torch.nn as nn
import torch.optim as optim


from torch.utils.data import dataloader
import dataset
import config
import my_utils
import models
import os
import time
from sklearn.metrics import f1_score
from torchsummary import summary
#print(torch.__version__) 1.13.1+cu117
import csv

def evaluate(bottom_model, top_model, loader):
    bottom_model.eval()
    top_model.eval()

    correct = 0
    num_samples = 0
    lossList, accList, F1List = [], [], []
    with torch.no_grad():
        for i, data in enumerate(loader):
            # 将数据加载到GPU上
            feature, labels = data[:, :-1], data[:, -1]
            feature = feature.to(config.device).float()
            labels = labels.to(config.device).long()
            output_bottom = bottom_model(feature)
            output_top = top_model(output_bottom)

            # 计算整体模型的输出
            probs = torch.softmax(output_top, dim=1)
            pred = torch.log_softmax(output_top, dim=1)
            loss = loss_object(pred, labels)

            correct += (probs.argmax(axis=1) == labels).type(torch.float).sum().item()
            num_samples += len(data)
            lossList.append(loss.cpu().detach().numpy())

            labels = labels.detach().cpu().numpy()
            pred = probs.argmax(1).detach().cpu().numpy()
            F1 = f1_score(labels, pred, average='micro')
            F1List.append(F1)

    acc = correct * 1.0/num_samples
    loss_test = numpy.mean(lossList)
    F1_test = numpy.mean(F1List)
    return loss_test, acc, F1_test


if __name__ == '__main__':
    start_time = time.time()

    # 配置对象
    config = config.Config()
    print(config.device)
    # 分别加载三个机构的训练和测试数据集
    trainDataset = dataset.Cifar10Dataset(config, institution='b', Flag='train')
    valDataset = dataset.Cifar10Dataset(config, institution='b', Flag='val')
    testDataset = dataset.Cifar10Dataset(config, institution='b', Flag='test')

    # 分别装载训练数据集和测试数据集
    train_loader = dataloader.DataLoader(
        dataset=trainDataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workders
    )
    val_loader = dataloader.DataLoader(
        dataset=valDataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workders
    )

    test_loader = dataloader.DataLoader(
        dataset=testDataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workders
    )
    # 创建3个模型，其中两个为passive_model, 一个为active_model
    bottom_model = models.Cifar10_BottomModel(in_channels=2, out_channels=6, kernel_size=5, hidden_size=200).to(
        config.device)
    top_model = models.Cifar10_TopModel(hidden_size=200, num_classes=10).to(config.device)

    # 优化目标及优化器
    loss_object = nn.NLLLoss()

    # 构建优化器
    optimizer = optim.Adam(list(bottom_model.parameters()) + list(top_model.parameters()),
                           lr=config.learning_rate_cifar10, weight_decay=0.001)  # weight_decay用于缓解过拟合现象

    # 模型训练过程
    best_acc = 0.0

    # 将训练过程中的loss写入文件
    with open('onlywithOnePart_cifar10_training_loss_2.csv', 'a+') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'epoch_loss'])

        for epoch in range(config.cifar10_epochs):
            bottom_model.train()
            top_model.train()

            lossList = []
            accList = []
            F1List = []

            for i, data in enumerate(train_loader):
                optimizer.zero_grad()
                feature, labels = data[:, :-1], data[:, -1]
                feature = feature.to(config.device).float()
                labels = labels.to(config.device).long()
                output_bottom = bottom_model(feature)
                output_top = top_model(output_bottom)
                # 机构C拥有标签
                pred_c = torch.log_softmax(output_top, dim=1)
                loss = loss_object(pred_c, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(bottom_model.parameters(), max_norm=3)
                nn.utils.clip_grad_norm_(top_model.parameters(), max_norm=3)
                optimizer.step()  # 更新model的参数

                probs = torch.softmax(output_top, dim=1)
                # correct = (probs.argmax(1) == labels.argmax(1)).type(torch.float).sum().item()
                correct = (probs.argmax(1) == labels).type(torch.float).sum().item()
                lossList.append(loss.cpu().detach().numpy())
                accList.append(correct / len(data))

                labels = labels.detach().cpu().numpy()
                pred = probs.argmax(1).detach().cpu().numpy()
                F1 = f1_score(labels, pred, average='micro')
                F1List.append(F1)


            # 每训练几轮则评测一下模型在验证集上的效果
            if (epoch + 1) % config.test_every == 0:
                loss_val, acc_val, F1_val = evaluate(bottom_model, top_model, val_loader)

            # 每一轮的结束 打印下该轮次得到的平均loss值和统计指标
            loss_epoch = numpy.mean(lossList)
            acc_epoch = numpy.mean(accList)
            F1_epoch = numpy.mean(F1List)
            writer.writerow([epoch, loss_epoch])
            template = 'EPOCH {}, Loss:{}, Accuracy:{}, F1:{}, ValLoss:{}, Val_Accuracy:{}, Val_F1:{}'
            print(template.format(epoch + 1,
                                  loss_epoch,
                                  acc_epoch * 100,
                                  F1_epoch*100,
                                  loss_val,
                                  acc_val * 100,
                                  F1_val*100))
            if acc_val > best_acc:
                best_acc = acc_val
                # 保存模型
                print('save--model--epoch-{}'.format(epoch + 1))
                torch.save(bottom_model, os.path.join(config.saved_dir_cifar10, 'onePartBottomModel.pth'))
                torch.save(top_model, os.path.join(config.saved_dir_cifar10, 'onePartTopModel.pth'))

    end_time = time.time()
    print('training process druing time:%f' % (end_time - start_time))

    start_time = time.time()

    # 训练结束之后，加载表现最好的模型
    bottom_model = torch.load(os.path.join(config.saved_dir_cifar10, 'onePartBottomModel.pth'))
    top_model = torch.load(os.path.join(config.saved_dir_cifar10, 'onePartTopModel.pth'))

    # 评估模型在测试集上的表现
    loss_test, acc_test, F1_test = evaluate(bottom_model, top_model, test_loader)
    template = 'Best TestLoss:{}, Test_Accuracy:{}, Test_F1:{}'
    print(template.format(loss_test, acc_test * 100, F1_test*100))
    end_time = time.time()
    print('Testing process druing time:%f' % (end_time - start_time))