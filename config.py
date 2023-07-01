#! usr/bin/env python 39
# -*- coding: utf-8 -*- 
# @Time  : 2023/3/29 10:51
# @Author: sharonswang   
# @File  : config.py
import torch

class Config(object):
	#data_path = './mnist.npz'
	adult_data_path = '/home/nbic/wangshanshan2023/unsplitting_additiveVFL/data/adult'
	cifar10_data_path = '/home/nbic/wangshanshan2023/additiveVFL/data/cifar-10-python'
	kc_house_price_datapath = '/home/nbic/wangshanshan2023/unsplitting_additiveVFL/data/kc-house-price'
	song_datapath = '/home/nbic/wangshanshan2023/unsplitting_additiveVFL/data/song'
	image_size = 28

	num_workders = 1
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	paticipantsNum = 2  # 候选值为2,3,4,5，默认为2

	# 训练时需要的参数
	batch_size = 64
	#learning_rate = 10e-5
	learning_rate_adult = 0.000001
	learning_rate_cifar10 = 0.00001
	learning_rate_kchouse = 0.000001
	learning_rate_song = 0.0001

	adult_epochs = 1000
	cifar10_epochs = 1000
	kchouse_epochs = 2000
	song_epochs = 2000
	test_every = 1

	saved_dir_adult = '/home/nbic/wangshanshan2023/unsplitting_additiveVFL/savedModels/adult'
	saved_dir_cifar10 = '/home/nbic/wangshanshan2023/additiveVFL/savedModels/cifar-10-python'
	saved_dir_kchouse = '/home/nbic/wangshanshan2023/additiveVFL/savedModels/kchouse'
	saved_dir_song = '/home/nbic/wangshanshan2023/additiveVFL/savedModels/song'



