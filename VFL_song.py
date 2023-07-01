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
import models
import os
import sklearn
import sklearn.metrics
import time
#print(torch.__version__) 1.13.1+cu117
import csv

def evaluate(bottom_model_a, bottom_model_b, top_model, loader_a, loader_b):
    bottom_model_a.eval()
    bottom_model_b.eval()
    top_model.eval()

    lossList, r2scoreList = [], []
    #with torch.no_grad():
    for i, data in enumerate(zip(loader_a, loader_b)):
        sample_a, sample_b = data[0], data[1]
        # 将数据加载到GPU上
        sample_a = sample_a.to(config.device).float()
        b_feature, b_labels = sample_b[:, :-1], sample_b[:, -1]
        b_feature = b_feature.to(config.device).float()
        b_labels = b_labels.to(config.device).float().view(-1, 1)

        output_bottom_b = bottom_model_b(b_feature)
        output_bottom_a = bottom_model_a(sample_a)
        input_top_model_b = torch.tensor([], requires_grad=True)
        input_top_model_a = torch.tensor([], requires_grad=True)
        input_top_model_b.data = output_bottom_b.data
        input_top_model_a.data = output_bottom_a.data
        input_top_model = torch.cat([input_top_model_b, input_top_model_a], dim=1)
        output_top = top_model(input_top_model)

        # 机构C拥有标签

        loss = loss_object(output_top, b_labels)
        mse_loss = torch.sqrt(loss)
        lossList.append(mse_loss.cpu().detach().numpy())
        r2_score = sklearn.metrics.r2_score(b_labels.detach().cpu().numpy(), output_top.detach().cpu().numpy())
        r2scoreList.append(r2_score)
    loss_test = numpy.mean(lossList)
    r2score_test = numpy.mean(r2scoreList)
    return loss_test, r2score_test


def up_date_parameters(optimizer, model, loss):

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
    optimizer.step()

if __name__ == '__main__':
    start_time = time.time()
    # 配置对象
    config = config.Config()
    print(config.device)

    # 分别加载三个机构的训练和测试数据集
    trainDataset_a = dataset.SongDataset(config, institution='a', Flag='train')
    valDataset_a = dataset.SongDataset(config, institution='a', Flag='val')
    testDataset_a = dataset.SongDataset(config, institution='a', Flag='test')
    trainDataset_b = dataset.SongDataset(config, institution='b', Flag='train')
    valDataset_b = dataset.SongDataset(config, institution='b', Flag='val')
    testDataset_b = dataset.SongDataset(config, institution='b', Flag='test')


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
    bottom_model_a = models.BottomModel(in_size=7, hidden_size=200).to(config.device)
    bottom_model_b = models.BottomModel(in_size=6, hidden_size=200).to(config.device)
    top_model = models.TopModel(hidden_size=400, num_classes=1).to(config.device)

    # 优化目标及优化器
    loss_object = nn.MSELoss()

    # 使用不同的优化器
    optimizer_bottom_b = optim.Adam(bottom_model_b.parameters(), lr=config.learning_rate_song)
    optimizer_bottom_a = optim.Adam(bottom_model_a.parameters(), lr=config.learning_rate_song)
    optimizer_top = optim.Adam(top_model.parameters(), lr=config.learning_rate_song)


    # 模型训练过程
    best_mse = 99999999
    # 将训练过程中的loss写入文件
    with open('VFL_song_training_loss.csv', 'a+') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'epoch_loss'])

        for epoch in range(config.song_epochs):
            bottom_model_b.train()
            bottom_model_a.train()
            top_model.train()

            lossList = []
            r2ScoreList = []

            for i, data in enumerate(zip(train_loader_a, train_loader_b)):

                sample_a, sample_b = data[0], data[1]
                # 将数据加载到GPU上
                sample_a = sample_a.to(config.device).float()
                b_feature, b_labels = sample_b[:, :-1], sample_b[:, -1]
                b_feature = b_feature.to(config.device).float()
                b_labels = b_labels.to(config.device).float().view(-1, 1)

                output_bottom_b = bottom_model_b(b_feature)
                output_bottom_a =bottom_model_a(sample_a)
                input_top_model_b = torch.tensor([], requires_grad=True)
                input_top_model_a = torch.tensor([], requires_grad=True)
                input_top_model_b.data = output_bottom_b.data
                input_top_model_a.data = output_bottom_a.data
                input_top_model = torch.cat([input_top_model_b, input_top_model_a], dim=1)
                output_top = top_model(input_top_model)

                # 机构C拥有标签
                loss = loss_object(output_top, b_labels)

                # 更新顶层模型的参数
                up_date_parameters(optimizer_top, top_model, loss)

                # 更新底层模型的参数
                grad_output_bottom_model_b = input_top_model_b.grad
                grad_output_bottom_model_a = input_top_model_a.grad

                loss_a = torch.sum(output_bottom_a * grad_output_bottom_model_a)
                up_date_parameters(optimizer_bottom_a, bottom_model_a, loss_a)

                loss_b = torch.sum(output_bottom_b * grad_output_bottom_model_b)
                up_date_parameters(optimizer_bottom_b, bottom_model_b, loss_b)

                mse_loss = torch.sqrt(loss)
                lossList.append(mse_loss.cpu().detach().numpy())
                r2_score = sklearn.metrics.r2_score(b_labels.detach().cpu().numpy(), output_top.detach().cpu().numpy())
                r2ScoreList.append(r2_score)

            # 每训练几轮则评测一下模型在验证集上的效果
            if (epoch+1) % config.test_every == 0:
                loss_val, r2score_val = evaluate(bottom_model_a, bottom_model_b, top_model, val_loader_a, val_loader_b)

            # 每一轮的结束 打印下该轮次得到的平均loss值和统计指标
            loss_epoch = numpy.mean(lossList)
            r2score_epoch = numpy.mean(r2ScoreList)
            writer.writerow([epoch, loss_epoch])
            template = 'EPOCH {}, Train_Loss:{}, Train_R2Score:{}, ValLoss:{}, Val_R2Score:{}'
            print(template.format(epoch + 1,
                                  loss_epoch,
                                  r2score_epoch,
                                  loss_val,
                                  r2score_val))

            if loss_val < best_mse:
                best_mse = loss_val
                # 保存模型
                print('save--model--epoch-{}'.format(epoch + 1))
                torch.save(bottom_model_a, os.path.join(config.saved_dir_song, 'bottom_model_a_VFL_2parts.pth'))
                torch.save(bottom_model_b, os.path.join(config.saved_dir_song, 'bottom_model_b_VFL_2parts.pth'))
                torch.save(top_model, os.path.join(config.saved_dir_song, 'top_model_VFL_2parts.pth'))

    end_time = time.time()
    print('training process druing time:%f' % (end_time - start_time))

    start_time = time.time()

    # 训练结束之后，加载表现最好的模型
    bottom_model_a = torch.load(os.path.join(config.saved_dir_song, 'bottom_model_a_VFL_2parts.pth'))
    bottom_model_b = torch.load(os.path.join(config.saved_dir_song, 'bottom_model_b_VFL_2parts.pth'))
    top_model = torch.load(os.path.join(config.saved_dir_song, 'top_model_VFL_2parts.pth'))


    # 评估模型在测试集上的表现
    loss_test, r2score_test = evaluate(bottom_model_a, bottom_model_b, top_model, test_loader_a, test_loader_b)
    template = 'Best TestLoss:{}, Test_R2score:{}'
    print(template.format(loss_test, r2score_test))
    end_time = time.time()
    print('testing process druing time:%f' % (end_time - start_time))
