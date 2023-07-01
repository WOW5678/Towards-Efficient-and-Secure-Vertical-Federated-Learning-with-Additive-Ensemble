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
from torchsummary import summary
from sklearn.metrics import f1_score
import csv
#print(torch.__version__) 1.13.1+cu117

def evaluate(bottom_model_a, bottom_model_b, top_model, loader_a, loader_b):
    bottom_model_a.eval()
    bottom_model_b.eval()
    top_model.eval()

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

        output_bottom_b = bottom_model_b(b_feature)
        output_bottom_a = bottom_model_a(sample_a)
        input_top_model_b = torch.tensor([], requires_grad=True)
        input_top_model_a = torch.tensor([], requires_grad=True)
        input_top_model_b.data = output_bottom_b.data
        input_top_model_a.data = output_bottom_a.data
        input_top_model = torch.cat([input_top_model_b, input_top_model_a], dim=1)
        output_top = top_model(input_top_model)

        # 机构C拥有标签
        pred = torch.log_softmax(output_top, dim=1)
        loss = loss_object(pred, b_labels)
        top_model.zero_grad()
        loss.backward()

        # 更新底层模型的参数
        grad_output_bottom_model_b = input_top_model_b.grad
        grad_output_bottom_model_a = input_top_model_a.grad

        loss_a = torch.sum(output_bottom_a * grad_output_bottom_model_a)
        up_date_parameters(optimizer_bottom_a, bottom_model_a, loss_a)

        loss_b = torch.sum(output_bottom_b * grad_output_bottom_model_b)
        up_date_parameters(optimizer_bottom_b, bottom_model_b, loss_b)

        soft_pred = torch.softmax(output_top, dim=1)
        correct = correct + (soft_pred.argmax(1) == b_labels).type(torch.float).sum().item()
        num_samples = num_samples + len(sample_a)
        lossList.append(loss.cpu().detach().numpy())

        b_labels = b_labels.detach().cpu().numpy()
        pred = soft_pred.argmax(1).detach().cpu().numpy()
        # print('b_labels:', b_labels)
        # print('pred:', pred)
        F1 = f1_score(b_labels, pred)
        F1List.append(F1)

    acc = correct * 1.0/num_samples
    loss_test = numpy.mean(lossList)
    F1_test = numpy.mean(F1List)
    return loss_test, acc, F1_test


def up_date_parameters(optimizer, model, loss):

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
    optimizer.step()

def direct_attack_label_on_gradient(bottom_model_grad, target_labels):
    attack_correct = 0
    for sample_id in range(len(bottom_model_grad)):
        grad_per_sample = bottom_model_grad[sample_id]
        for logit_id in range(len(grad_per_sample)):
            if grad_per_sample[logit_id] < 0:
                inferred_label = logit_id
                if inferred_label == target_labels[sample_id]:
                    attack_correct += 1
                    break

    return attack_correct

if __name__ == '__main__':
    start_time = time.time()

    # 配置对象
    config = config.Config()
    print(config.device)
    # 分别加载三个机构的训练和测试数据集
    trainDataset_a = dataset.AdultDataset(config, institution='a', Flag='train')
    valDataset_a = dataset.AdultDataset(config, institution='a', Flag='val')
    testDataset_a = dataset.AdultDataset(config, institution='a', Flag='test')
    trainDataset_b = dataset.AdultDataset(config, institution='b', Flag='train')
    valDataset_b = dataset.AdultDataset(config, institution='b', Flag='val')
    testDataset_b = dataset.AdultDataset(config, institution='b', Flag='test')


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

    # 创建2个模型，其中两个为passive_model, 一个为active_model
    bottom_model_a = models.BottomModel(in_size=50, hidden_size=200).to(config.device)
    bottom_model_b = models.BottomModel(in_size=57, hidden_size=200).to(config.device)
    top_model = models.TopModel(hidden_size=400, num_classes=2).to(config.device)

    print('bottom_model_a:', summary(bottom_model_a, (1, 50)))
    print('bottom_model_b:', summary(bottom_model_b, (1, 57)))

    # 优化目标及优化器
    loss_object = nn.NLLLoss()

    # 使用不同的优化器
    optimizer_bottom_b = optim.Adam(bottom_model_b.parameters(), lr=config.learning_rate_adult)
    optimizer_bottom_a = optim.Adam(bottom_model_a.parameters(), lr=config.learning_rate_adult)
    optimizer_top = optim.Adam(top_model.parameters(), lr=config.learning_rate_adult)


    # 模型训练过程
    best_acc = 0.0
    # 将训练过程中的loss写入文件
    with open('VFL_adult_training_loss_3.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'epoch_loss'])

        for epoch in range(config.adult_epochs):
            bottom_model_b.train()
            bottom_model_a.train()
            top_model.train()

            lossList = []
            accList = []
            F1List = []
            for i, data in enumerate(zip(train_loader_a, train_loader_b)):

                sample_a, sample_b = data[0], data[1]
                # 将数据加载到GPU上
                sample_a = sample_a.to(config.device).float()
                b_feature, b_labels = sample_b[:, :-1], sample_b[:, -1]
                b_feature = b_feature.to(config.device).float()
                b_labels = b_labels.to(config.device).long()

                output_bottom_b = bottom_model_b(b_feature)
                output_bottom_a =bottom_model_a(sample_a)
                input_top_model_b = torch.tensor([], requires_grad=True)
                input_top_model_a = torch.tensor([], requires_grad=True)
                input_top_model_b.data = output_bottom_b.data
                input_top_model_a.data = output_bottom_a.data
                input_top_model = torch.cat([input_top_model_b, input_top_model_a], dim=1)
                output_top = top_model(input_top_model)

                # 机构C拥有标签
                pred = torch.log_softmax(output_top, dim=1)
                loss = loss_object(pred, b_labels)

                # 更新顶层模型的参数
                up_date_parameters(optimizer_top, top_model, loss)

                # 更新底层模型的参数
                grad_output_bottom_model_b = input_top_model_b.grad
                grad_output_bottom_model_a = input_top_model_a.grad

                loss_a = torch.sum(output_bottom_a * grad_output_bottom_model_a)
                up_date_parameters(optimizer_bottom_a, bottom_model_a, loss_a)

                loss_b = torch.sum(output_bottom_b* grad_output_bottom_model_b)
                up_date_parameters(optimizer_bottom_b, bottom_model_b, loss_b)

                soft_pred = torch.softmax(output_top, dim=1)
                correct = (soft_pred.argmax(1) == b_labels).type(torch.float).sum().item()
                lossList.append(loss.cpu().detach().numpy())
                accList.append(correct/len(sample_a))

                b_labels = b_labels.detach().cpu().numpy()
                pred = soft_pred.argmax(1).detach().cpu().numpy()
                F1 = f1_score(b_labels, pred)
                F1List.append(F1)

            # 每训练几轮则评测一下模型在验证集上的效果
            if epoch % config.test_every == 0:
                loss_val, acc_val, F1_val = evaluate(bottom_model_a, bottom_model_b, top_model, val_loader_a, val_loader_b)

            # 每一轮的结束 打印下该轮次得到的平均loss值和统计指标
            loss_epoch = numpy.mean(lossList)
            acc_epoch = numpy.mean(accList)
            F1_epoch = numpy.mean(F1List)
            writer.writerow([epoch, loss_epoch])
            template = 'EPOCH {}, Loss:{}, Accuracy:{}, F1:{}, ValLoss:{}, Val_Accuracy:{}, Val_F1:{}'
            print(template.format(epoch + 1,
                                  loss_epoch,
                                  acc_epoch*100,
                                  F1*100,
                                  loss_val,
                                  acc_val*100,
                                  F1_val*100))

            if not os.path.exists(config.saved_dir_adult):
                os.makedirs(config.saved_dir_adult)

            if acc_val > best_acc:
                best_acc = acc_val
                # 保存模型
                torch.save(bottom_model_a, os.path.join(config.saved_dir_adult, 'bottom_model_a_VFL_2parts.pth'))
                torch.save(bottom_model_b, os.path.join(config.saved_dir_adult, 'bottom_model_b_VFL_2parts.pth'))
                torch.save(top_model, os.path.join(config.saved_dir_adult, 'top_model_VFL_2parts.pth'))

    end_time = time.time()
    print('training process druing time:%f' % (end_time - start_time))

    start_time = time.time()
    # 训练结束之后，加载表现最好的模型
    bottom_model_a = torch.load(os.path.join(config.saved_dir_adult, 'bottom_model_a_VFL_2parts.pth'))
    bottom_model_b = torch.load(os.path.join(config.saved_dir_adult, 'bottom_model_b_VFL_2parts.pth'))
    top_model = torch.load(os.path.join(config.saved_dir_adult, 'top_model_VFL_2parts.pth'))


    # 评估模型在测试集上的表现
    loss_test, acc_test, F1_test = evaluate(bottom_model_a, bottom_model_b, top_model, test_loader_a, test_loader_b)
    template = 'Best TestLoss:{}, Test_Accuracy:{}, Test_F1:{}'
    print(template.format(loss_test, acc_test * 100, F1_test*100))

    end_time = time.time()
    print('Testing process druing time:%f' % (end_time - start_time))