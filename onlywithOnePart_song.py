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
import sklearn
import sklearn.metrics
import time
import csv

#print(torch.__version__) 1.13.1+cu117

def evaluate(bottom_model, top_model, loader):
    bottom_model.eval()
    top_model.eval()

    lossList, r2ScoreList = [], []
    with torch.no_grad():
        for i, data in enumerate(loader):
            # 将数据加载到GPU上
            feature, labels = data[:, :-1], data[:, -1]
            feature = feature.to(config.device).float()
            labels = labels.to(config.device).float().view(-1, 1)
            output_bottom = bottom_model(feature)
            output_top = top_model(output_bottom)

            # 计算整体模型的输出
            loss = loss_object(output_top, labels)
            rmse_loss = torch.sqrt(loss)
            lossList.append(rmse_loss.cpu().detach().numpy())
            r2_score = sklearn.metrics.r2_score(labels.detach().cpu().numpy(), output_top.detach().cpu().numpy())
            r2ScoreList.append(r2_score)

    loss_test = numpy.mean(lossList)
    r2Score_test = numpy.mean(r2ScoreList)
    return loss_test, r2Score_test


if __name__ == '__main__':

    start_time = time.time()
    # 配置对象
    config = config.Config()
    print(config.device)
    # 分别加载三个机构的训练和测试数据集
    trainDataset = dataset.SongDataset(config, institution='b', Flag='train')
    valDataset = dataset.SongDataset(config, institution='b', Flag='val')
    testDataset = dataset.SongDataset(config, institution='b', Flag='test')

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

    bottom_model = models.RegressionBottomModel(in_size=6, hidden_size=200).to(config.device)
    top_model = models.RegressionTopModel(hidden_size=200, num_classes=1).to(config.device)

    # 优化目标及优化器
    loss_object = nn.MSELoss()

    # 构建优化器
    optimizer = optim.Adam(list(bottom_model.parameters()) + list(top_model.parameters()), lr=config.learning_rate_song, weight_decay=0.001) # weight_decay用于缓解过拟合现象

    # 模型训练过程
    best_loss = 999999999
    # 将训练过程中的loss写入文件
    with open('onlywithOnePart_song_training_loss.csv', 'a+') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'epoch_loss'])

        for epoch in range(config.song_epochs):
            bottom_model.train()
            top_model.train()

            lossList = []
            r2ScoreList = []
            for i, data in enumerate(train_loader):
                optimizer.zero_grad()
                feature, labels = data[:, :-1], data[:, -1]
                feature = feature.to(config.device).float()
                labels = labels.to(config.device).float().view(-1, 1)
                output_bottom = bottom_model(feature)
                output_top = top_model(output_bottom)
                # 机构B拥有标签
                loss = loss_object(output_top, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(bottom_model.parameters(), max_norm=3)
                nn.utils.clip_grad_norm_(top_model.parameters(), max_norm=3)
                optimizer.step()  # 更新model的参数

                rmse_loss = torch.sqrt(loss)
                lossList.append(rmse_loss.cpu().detach().numpy())
                r2_score = sklearn.metrics.r2_score(labels.detach().cpu().numpy(), output_top.detach().cpu().numpy())
                r2ScoreList.append(r2_score)

            # 每训练几轮则评测一下模型在验证集上的效果
            if (epoch+1) % config.test_every == 0:
                loss_val, r2Score_val = evaluate(bottom_model, top_model, val_loader)

            # 每一轮的结束 打印下该轮次得到的平均loss值和统计指标
            loss_epoch = numpy.mean(lossList)
            r2Score_epoch = numpy.mean(r2ScoreList)
            writer.writerow([epoch, loss_epoch])

            template = 'EPOCH {}, TrainLoss:{}, TrainR2score:{}, ValLoss:{}, ValR2score:{}'
            print(template.format(epoch + 1,
                                  loss_epoch,
                                  r2Score_epoch,
                                  loss_val,
                                  r2Score_val))
            if loss_val < best_loss:
                best_loss = loss_val
                # 保存模型
                print('save--model--epoch-{}'.format(epoch+1))
                torch.save(bottom_model, os.path.join(config.saved_dir_song, 'onePartBottomModel.pth'))
                torch.save(top_model, os.path.join(config.saved_dir_song, 'onePartTopModel.pth'))

    end_time = time.time()
    print('training process druing time:%f' % (end_time - start_time))

    start_time = time.time()

    # 训练结束之后，加载表现最好的模型
    bottom_model = torch.load(os.path.join(config.saved_dir_song, 'onePartBottomModel.pth'))
    top_model = torch.load(os.path.join(config.saved_dir_song, 'onePartTopModel.pth'))

    # 评估模型在测试集上的表现
    loss_test, r2Score_test = evaluate(bottom_model, top_model, test_loader)
    template = 'Best TestLoss:{}, Test_r2score:{}'
    print(template.format(loss_test, r2Score_test))

    end_time = time.time()
    print('testing process druing time:%f' % (end_time - start_time))

