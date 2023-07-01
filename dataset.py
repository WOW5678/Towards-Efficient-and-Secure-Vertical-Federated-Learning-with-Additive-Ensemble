#! usr/bin/env python 39
# -*- coding: utf-8 -*- 
# @Time  : 2023/3/29 10:49
# @Author: sharonswang   
# @File  : dataset.py

import torch
from torch.utils.data import Dataset
from numpy import load
import pandas as pd
import os

class MnistDataset(Dataset):
	def __init__(self, config, trainFlag):
		# 加载数据
		data = load(config.data_path)
		# print(data.files)  # 'x_test', 'x_train', 'y_train', 'y_test'
		if trainFlag:
			self.data_x = data['x_train']/255.0
			self.data_y = data['y_train']
		else:
			self.data_x = data['x_test']/255.0
			self.data_y = data['y_test']

		# 对y进行one-hot编码
		#self.data_y = self.int2onehot()

	def __getitem__(self, index):
		x = self.data_x[index]
		y = self.data_y[index]
		return x, y

	def __len__(self):
		return len(self.data_x)

	def int2onehot(self):
		nb_digits = 10
		y = torch.LongTensor(len(self.data_y), 1).random_() % nb_digits
		# One hot encoding buffer that you create out of the loop and just keep reusing
		y_onehot = torch.FloatTensor(len(self.data_y), nb_digits)

		y_onehot.zero_()
		y_onehot.scatter_(1, y, 1)
		y = torch.LongTensor(len(self.data_y), 1).random_() % nb_digits
		# One hot encoding buffer that you create out of the loop and just keep reusing
		y_onehot = torch.FloatTensor(len(self.data_y), nb_digits)
		return y_onehot


class AdultDataset(Dataset):
	def __init__(self, config, institution, Flag):
		if config.paticipantsNum == 2:
			datadirs = os.path.join('/home/nbic/wangshanshan2023/additiveVFL/data/adult', 'twoPaticipants')
		elif config.paticipantsNum == 3:
			datadirs = os.path.join('/home/nbic/wangshanshan2023/additiveVFL/data/adult', 'threePaticipants')
		elif config.paticipantsNum == 4:
			datadirs = os.path.join('/home/nbic/wangshanshan2023/additiveVFL/data/adult', 'fourPaticipants')
		elif config.paticipantsNum == 5:
			datadirs = os.path.join('/home/nbic/wangshanshan2023/additiveVFL/data/adult', 'fivePaticipants')
		self.institution = institution
		# 加载数据集
		if Flag == 'train':
			data_path = os.path.join(datadirs, 'train_%s.npz'%self.institution)
			data = load(data_path)
			# print('data:', data.files) #
			self.data = data['arr_0']
			print(self.data.shape)
		elif Flag == 'val':
			data_path = os.path.join(datadirs, 'val_%s.npz'%self.institution)
			data = load(data_path)
			# print('data:', data.files)
			self.data = data['arr_0']
			print(self.data.shape)

		elif Flag =='test':
			data_path = os.path.join(datadirs, 'test_%s.npz' % self.institution)
			data = load(data_path)
			# print('data:', data.files)
			self.data = data['arr_0']
			print(self.data.shape)

	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)

class FullAdultDataset(Dataset):
	def __init__(self, config, Flag):
		# 加载数据集
		if Flag == 'train':
			data_path = os.path.join('data/adult', 'adult.train.npz')
			data = load(data_path)
			# print('data:', data.files) #
			self.data = data['arr_0']
			print(self.data.shape)
		elif Flag == 'val':
			data_path = os.path.join('data/adult', 'adult.val.npz')
			data = load(data_path)
			# print('data:', data.files)
			self.data = data['arr_0']
			print(self.data.shape)

		elif Flag == 'test':
			data_path = os.path.join('data/adult', 'adult.test.npz')
			data = load(data_path)
			# print('data:', data.files)
			self.data = data['arr_0']
			print(self.data.shape)



	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)

class Cifar10Dataset(Dataset):
	def __init__(self, config, institution, Flag):

		if config.paticipantsNum == 2:
			datadirs = os.path.join(config.cifar10_data_path, 'twoPaticipants')
		elif config.paticipantsNum == 3:
			datadirs = os.path.join(config.cifar10_data_path, 'threePaticipants')
		elif config.paticipantsNum == 4:
			datadirs = os.path.join(config.cifar10_data_path, 'fourPaticipants')
		elif config.paticipantsNum == 5:
			datadirs = os.path.join(config.cifar10_data_path, 'fivePaticipants')
		self.institution = institution

		# 加载数据集
		if Flag == 'train':
			data_path = os.path.join(datadirs, 'train_%s.npz' % self.institution)
			data = load(data_path)
			# print('data:', data.files) #
			self.data = data['arr_0']
			print('train:', self.data.shape)
		elif Flag == 'val':
			data_path = os.path.join(datadirs, 'val_%s.npz' % self.institution)
			data = load(data_path)
			# print('data:', data.files)
			self.data = data['arr_0']
			print('val:', self.data.shape)

		elif Flag =='test':
			data_path = os.path.join(datadirs, 'test_%s.npz' % self.institution)
			data = load(data_path)
			# print('data:', data.files)
			self.data = data['arr_0']
			print('test:', self.data.shape)

	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)


class FullCifar10Dataset(Dataset):
	def __init__(self, config, Flag):
		# 加载数据集
		if Flag == 'train':
			data_path = os.path.join(config.cifar10_data_path, 'cifar10.train.npz')
			data = load(data_path)
			# print('data:', data.files) #
			self.data = data['arr_0']
			print('train:', self.data.shape)
		elif Flag == 'val':
			data_path = os.path.join(config.cifar10_data_path, 'cifar10.val.npz')
			data = load(data_path)
			# print('data:', data.files)
			self.data = data['arr_0']
			print('val:', self.data.shape)

		elif Flag == 'test':
			data_path = os.path.join(config.cifar10_data_path, 'cifar10.test.npz')
			data = load(data_path)
			# print('data:', data.files)
			self.data = data['arr_0']
			print('test', self.data.shape)


	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)


class KCHouseDataset(Dataset):
	def __init__(self, config, institution, Flag):

		if config.paticipantsNum == 2:
			datadirs = os.path.join(config.kc_house_price_datapath, 'twoPaticipants')
		elif config.paticipantsNum == 3:
			datadirs = os.path.join(config.kc_house_price_datapath, 'threePaticipants')
		elif config.paticipantsNum == 4:
			datadirs = os.path.join(config.kc_house_price_datapath, 'fourPaticipants')
		elif config.paticipantsNum == 5:
			datadirs = os.path.join(config.kc_house_price_datapath, 'fivePaticipants')
		self.institution = institution

		# 加载数据集
		if Flag == 'train':
			data_path = os.path.join(datadirs, 'train_%s.npz' % self.institution)
			data = load(data_path)
			# print('data:', data.files) #
			self.data = data['arr_0']
			print('train:',self.data.shape)
		elif Flag == 'val':
			data_path = os.path.join(datadirs, 'val_%s.npz' % self.institution)
			data = load(data_path)
			# print('data:', data.files)
			self.data = data['arr_0']
			print('val:', self.data.shape)

		elif Flag =='test':
			data_path = os.path.join(datadirs, 'test_%s.npz' % self.institution)
			data = load(data_path)
			# print('data:', data.files)
			self.data = data['arr_0']
			print('test:', self.data.shape)

	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)


class FullKCHouseDataset(Dataset):
	def __init__(self, config, Flag):
		# 加载数据集
		if Flag == 'train':
			data_path = os.path.join(config.kc_house_price_datapath, 'kchouse.train.npz')
			data = load(data_path)
			# print('data:', data.files) #
			self.data = data['arr_0']
			print(self.data.shape)
		elif Flag == 'val':
			data_path = os.path.join(config.kc_house_price_datapath, 'kchouse.val.npz')
			data = load(data_path)
			# print('data:', data.files)
			self.data = data['arr_0']
			print(self.data.shape)

		elif Flag == 'test':
			data_path = os.path.join(config.kc_house_price_datapath, 'kchouse.test.npz')
			data = load(data_path)
			# print('data:', data.files)
			self.data = data['arr_0']
			print(self.data.shape)


	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)

class SongDataset(Dataset):
	def __init__(self, config, institution, Flag):

		if config.paticipantsNum == 2:
			datadirs = os.path.join(config.song_datapath, 'twoPaticipants')
		elif config.paticipantsNum == 3:
			datadirs = os.path.join(config.song_datapath, 'threePaticipants')
		elif config.paticipantsNum == 4:
			datadirs = os.path.join(config.song_datapath, 'fourPaticipants')
		elif config.paticipantsNum == 5:
			datadirs = os.path.join(config.song_datapath, 'fivePaticipants')
		self.institution = institution

		# 加载数据集
		if Flag == 'train':
			data_path = os.path.join(datadirs, 'train_%s.npz' % self.institution)
			data = load(data_path)
			# print('data:', data.files) #
			self.data = data['arr_0']
			print('train:',self.data.shape)
		elif Flag == 'val':
			data_path = os.path.join(datadirs, 'val_%s.npz' % self.institution)
			data = load(data_path)
			# print('data:', data.files)
			self.data = data['arr_0']
			print('val:', self.data.shape)

		elif Flag =='test':
			data_path = os.path.join(datadirs, 'test_%s.npz' % self.institution)
			data = load(data_path)
			# print('data:', data.files)
			self.data = data['arr_0']
			print('test:', self.data.shape)

	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)


class FullSongDataset(Dataset):
	def __init__(self, config, Flag):
		# 加载数据集
		if Flag == 'train':
			data_path = os.path.join(config.song_datapath, 'song.train.npz')
			data = load(data_path)
			# print('data:', data.files) #
			self.data = data['arr_0']
			print(self.data.shape)
		elif Flag == 'val':
			data_path = os.path.join(config.song_datapath, 'song.val.npz')
			data = load(data_path)
			# print('data:', data.files)
			self.data = data['arr_0']
			print(self.data.shape)

		elif Flag == 'test':
			data_path = os.path.join(config.song_datapath, 'song.test.npz')
			data = load(data_path)
			# print('data:', data.files)
			self.data = data['arr_0']
			print(self.data.shape)


	def __getitem__(self, index):
		return self.data[index]

	def __len__(self):
		return len(self.data)
