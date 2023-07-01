# -*- coding: utf-8 -*-
#! usr/bin/env python 39
# @Time  : 2023/3/23 13:26
# @Author: sharonswang
# @File  : main.py
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import os

from torch.utils.data import dataloader
import sys
sys.path.extend('F:\\Research\\additiveVFL')
from additiveVFL import dataset
from additiveVFL import config
from additiveVFL import my_utils
from additiveVFL import models
from sklearn.metrics import f1_score
import time
from torchsummary import summary
# print(torch.__version__)  #1.13.1+cu117
import csv

def evaluate(bottom_model_a, bottom_model_b, top_model_a, top_model_b, loader_a, loader_b):
    bottom_model_a.eval()
    top_model_a.eval()
    bottom_model_b.eval()
    top_model_b.eval()

    correct = 0
    num_samples = 0
    lossList, accList, F1List = [], [], []
    #with torch.no_grad():
    for i, data in enumerate(zip(loader_a, loader_b)):
        sample_a, sample_b = data[0], data[1]

        # 将数据加载到GPU上
        sample_a = sample_a.to(config.device).float()
        b_feature, b_labels = sample_b[:, :-1], sample_b[:, -1]
        b_feature = b_feature.to(config.device).float()
        b_labels = b_labels.to(config.device).long()

        # 机构B

        output_b = bottom_model_b(b_feature)
        input_top_model_b = torch.tensor([], requires_grad=True)
        input_top_model_b.data = output_b.data
        F1 = top_model_b(input_top_model_b)
        p1 = torch.softmax(F1, dim=1)
        pred_1 = torch.log(p1 + 1e-10)
        loss_1 = loss_object(pred_1, b_labels)
        optimizer_top_b.zero_grad()
        loss_1.backward()
        grad_output_bottom_model_b = input_top_model_b.grad
        loss_bottom_b = torch.sum(grad_output_bottom_model_b * output_b)

        # 机构A不拥有标签

        
        F1 = F1.detach()
        output_bottom_a = bottom_model_a(sample_a)
        input_top_model_a = torch.tensor([], requires_grad=True)
        input_top_model_a.data = output_bottom_a.data
        output_top_a = top_model_a(input_top_model_a)
        b_labels_onehot = my_utils.int2onehot(config, b_labels, num_classes=10)
        row_a = drive_weight(b_labels_onehot, F1, output_top_a)
        F2 = F1 + output_top_a * row_a
        p2 = torch.softmax(F2, dim=1)
        pred_2 = torch.log(p2 + 1e-10)
        loss_2 = loss_object(pred_2, b_labels)

        optimizer_top_b.zero_grad()
        loss_2.backward()
        grad_output_bottom_model_a = input_top_model_a.grad
        loss_bottom_a = torch.sum(grad_output_bottom_model_a * output_bottom_a)

        correct = correct + (p2.argmax(axis=1) == b_labels).type(torch.float).sum().item()
        num_samples = num_samples + len(sample_a)
        lossList.append(loss_2.cpu().detach().numpy())

        b_labels = b_labels.detach().cpu().numpy()
        pred = p2.argmax(1).detach().cpu().numpy()
        F1 = f1_score(b_labels, pred, average='micro')
        F1List.append(F1)

    acc = correct * 1.0/num_samples
    loss_test = numpy.mean(lossList)
    F1_test = numpy.mean(F1List)
    return loss_test, acc, F1_test

def drive_weight(y, F, h):
    p = torch.exp(F - torch.logsumexp(F, axis=1, keepdims=True))
    residuals = y - torch.where(torch.isnan(p), torch.full_like(p, 0), p)

    row_numerator = torch.sum(torch.multiply(h, residuals), dim=1)
    row_denominator = torch.multiply(torch.pow(h, 2), p)
    row_denominator = torch.sum(torch.multiply(row_denominator, p - 1), dim=1)
    row = - (row_numerator / (row_denominator+1e-10)).view(-1, 1)  # [batch_size, 1]
    return row

def up_date_parameters(optimizer, model, loss):
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
    optimizer.step()
    return

def init_network(net):
    # 初始化网络
    for m in net.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
            torch.nn.init.xavier_uniform_(m.weight)

def random_all():
    pass

if __name__ == '__main__':
    start_time = time.time()

    # 配置对象
    config = config.Config()
    print(config.device)
    # 分别加载三个机构的训练和测试数据集
    trainDataset_a = dataset.Cifar10Dataset(config, institution='a', Flag='train')
    valDataset_a = dataset.Cifar10Dataset(config, institution='a', Flag='val')
    testDataset_a = dataset.Cifar10Dataset(config, institution='a', Flag='test')
    trainDataset_b = dataset.Cifar10Dataset(config, institution='b', Flag='train')
    valDataset_b = dataset.Cifar10Dataset(config, institution='b', Flag='val')
    testDataset_b = dataset.Cifar10Dataset(config, institution='b', Flag='test')

    # 分别装载三个机构的数据集
    train_loader_a = dataloader.DataLoader(
        dataset=trainDataset_a,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workders
    )
    val_loader_a = dataloader.DataLoader(
        dataset=valDataset_a,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workders
    )
    test_loader_a = dataloader.DataLoader(
        dataset=testDataset_a,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workders
    )

    train_loader_b = dataloader.DataLoader(
        dataset=trainDataset_b,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workders
    )
    val_loader_b = dataloader.DataLoader(
        dataset=valDataset_b,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workders
    )
    test_loader_b = dataloader.DataLoader(
        dataset=testDataset_b,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workders
    )

    # 创建4个模型，2个bottom模型和2个top模型
    bottom_model_a = models.Cifar10_BottomModel(in_channels=1, out_channels=6, kernel_size=5, hidden_size=200).to(config.device)
    bottom_model_b = models.Cifar10_BottomModel(in_channels=2, out_channels=6, kernel_size=5, hidden_size=200).to(config.device)
    top_model_a = models.Cifar10_TopModel(hidden_size=200, num_classes=10).to(config.device)
    top_model_b = models.Cifar10_TopModel(hidden_size=200, num_classes=10).to(config.device)

    print('bottom_model_a:', summary(bottom_model_a, (1, 32, 32)))
    print('bottom_model_b:', summary(bottom_model_b, (2, 32, 32)))

    # 实验证实加上这块模型参数初始化导致模型结果反而急剧下降
    # 初始化网络
    # init_network(bottom_model_b)
    # init_network(top_model_b)
    # init_network(bottom_model_a)
    # init_network(top_model_a)

    # 优化目标及优化器
    loss_object = nn.NLLLoss()

    # 使用不同的优化器优化不同的参数部分
    optimizer_bottom_a = optim.Adam(bottom_model_a.parameters(), lr=config.learning_rate_cifar10)
    optimizer_bottom_b = optim.Adam(bottom_model_b.parameters(), lr=config.learning_rate_cifar10)
    optimizer_top_a = optim.Adam(top_model_a.parameters(), lr=config.learning_rate_cifar10)
    optimizer_top_b = optim.Adam(top_model_b.parameters(), lr=config.learning_rate_cifar10)

    # 模型训练过程
    best_acc = 0.0
    # 将训练过程中的loss写入文件
    with open('additiveVFL_cifar10_training_loss_2.csv', 'a+') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'epoch_loss', 'epoch_loss_1', 'epoch_loss_2'])


        for epoch in range(config.cifar10_epochs):
            bottom_model_a.train()
            bottom_model_b.train()
            top_model_a.train()
            top_model_b.train()

            lossList = []
            accList = []
            F1List = []
            loss1List = []
            loss2List = []
            for i, data in enumerate(zip(train_loader_a, train_loader_b)):
                sample_a, sample_b = data[0], data[1]

                # 将数据加载到GPU上
                sample_a = sample_a.to(config.device).float()
                b_feature, b_labels = sample_b[:, :-1], sample_b[:, -1]
                b_feature = b_feature.to(config.device).float()
                b_labels = b_labels.to(config.device).long()

                # 机构B拥有标签

                output_b = bottom_model_b(b_feature)
                input_top_model_b = torch.tensor([], requires_grad=True)
                input_top_model_b.data = output_b.data
                F1 = top_model_b(input_top_model_b)
                p1 = torch.softmax(F1, dim=1)
                pred_1 = torch.log(p1 + 1e-10)
                loss_1 = loss_object(pred_1, b_labels)
                up_date_parameters(optimizer_top_b, top_model_b, loss_1)

                grad_output_bottom_model_b = input_top_model_b.grad
                # print('grad_output_bottom_model_b:', grad_output_bottom_model_b)
                loss_bottom_b = torch.sum(grad_output_bottom_model_b * output_b)

                # 更新bottom model b的参数
                up_date_parameters(optimizer_bottom_b, bottom_model_b, loss_bottom_b)

                # 机构A不拥有标签
                F1 = F1.detach()
                output_bottom_a = bottom_model_a(sample_a)
                input_top_model_a = torch.tensor([], requires_grad=True)
                input_top_model_a.data = output_bottom_a.data
                output_top_a = top_model_a(input_top_model_a)
                b_labels_onehot = my_utils.int2onehot(config, b_labels, num_classes=10).to(config.device)
                row_a = drive_weight(b_labels_onehot, F1, output_top_a)
                F2 = F1 + output_top_a * row_a
                p2 = torch.softmax(F2, dim=1)
                pred_2 = torch.log(p2 + 1e-10)
                loss_2 = loss_object(pred_2, b_labels)
                up_date_parameters(optimizer_top_a, top_model_a, loss_2)

                grad_output_bottom_model_a = input_top_model_a.grad
                loss_bottom_a = torch.sum(grad_output_bottom_model_a * output_bottom_a)
                up_date_parameters(optimizer_bottom_a, bottom_model_a, loss_bottom_a)

                correct = (p2.argmax(1) == b_labels).type(torch.float).sum().item()
                lossList.append(loss_2.cpu().detach().numpy())
                accList.append(correct / len(sample_a))

                b_labels = b_labels.detach().cpu().numpy()
                pred = p2.argmax(1).detach().cpu().numpy()
                F1 = f1_score(b_labels, pred, average='micro')
                F1List.append(F1)

                loss1List.append(loss_1.item())
                loss2List.append(loss_2.item())

            # 每训练几轮则评测一下模型在验证集上的效果
            if epoch % config.test_every == 0:
                loss_val, acc_val, F1_val = evaluate(bottom_model_a, bottom_model_b, top_model_a, top_model_b, val_loader_a, val_loader_b)

            # 每一轮的结束 打印下该轮次得到的平均loss值和统计指标
            loss_epoch = numpy.mean(lossList)
            loss1_epoch = numpy.mean(loss1List)
            loss2_epoch = numpy.mean(loss2List)
            acc_epoch = numpy.mean(accList)
            F1_epoch = numpy.mean(F1List)
            template = 'EPOCH {}, Loss:{}, Accuracy:{}, F1:{}, ValLoss:{}, Val_Accuracy:{}, Val_F1:{}'
            writer.writerow([epoch, loss_epoch, loss1_epoch, loss2_epoch])
            print(template.format(epoch + 1,
                                  loss_epoch,
                                  acc_epoch*100,
                                  F1_epoch*100,
                                  loss_val,
                                  acc_val*100,
                                  F1_val*100))
            if not os.path.exists(config.saved_dir_cifar10):
                os.makedirs(config.saved_dir_cifar10)

            if acc_val > best_acc:
                best_acc = acc_val
                # 保存模型
                print('save--model--epoch-{}'.format(epoch + 1))
                torch.save(bottom_model_b, os.path.join(config.saved_dir_cifar10, 'bottom_model_b_2parts.pth'))
                torch.save(top_model_b, os.path.join(config.saved_dir_cifar10, 'top_model_b_2parts.pth'))
                torch.save(bottom_model_a, os.path.join(config.saved_dir_cifar10, 'bottom_model_a_2parts.pth'))
                torch.save(top_model_a, os.path.join(config.saved_dir_cifar10, 'top_model_a_2parts.pth'))

    end_time = time.time()
    print('training process druing time:%f' % (end_time - start_time))

    start_time = time.time()
    # 训练结束之后，加载表现最好的模型
    bottom_model_b = torch.load(os.path.join(config.saved_dir_cifar10, 'bottom_model_b_2parts.pth'))
    top_model_b = torch.load(os.path.join(config.saved_dir_cifar10, 'top_model_b_2parts.pth'))
    bottom_model_a = torch.load(os.path.join(config.saved_dir_cifar10, 'bottom_model_a_2parts.pth'))
    top_model_a = torch.load(os.path.join(config.saved_dir_cifar10, 'top_model_a_2parts.pth'))

    # 评估模型在测试集上的表现
    loss_test, acc_test, F1_test = evaluate(bottom_model_a, bottom_model_b, top_model_a, top_model_b, test_loader_a, test_loader_b)
    template = 'Best TestLoss:{}, Test_Accuracy:{}, Test_F1:{}'
    print(template.format(loss_test, acc_test * 100, F1_test*100))
    end_time = time.time()
    print('Testing process druing time:%f' % (end_time - start_time))
