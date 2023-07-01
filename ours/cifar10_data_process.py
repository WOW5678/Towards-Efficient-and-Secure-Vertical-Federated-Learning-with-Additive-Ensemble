#! usr/bin/env python 39
# -*- coding: utf-8 -*- 
# @Time  : 2023/4/18 11:10
# @Author: sharonswang   
# @File  : cifar10_data_process.py

import numpy as np
import random
import pickle
import platform
import os
from sklearn.model_selection import train_test_split

# 加载序列文件

def load_pickle(f):
	version = platform.python_version_tuple()  # 判断python的版本
	if version[0] == '2':
		return pickle.load(f)
	elif version[0] == '3':
		return pickle.load(f, encoding='latin1')
	raise ValueError('invalid python version:{}'.format(version))

def load_cifar_batch(filename):
	with open(filename, 'rb') as f:
		datadict = load_pickle(f)
		x = datadict['data']
		y = datadict['labels']
		print(x.shape)
		x = x.reshape(-1, 3* 32* 32).astype('float32')
		y = np.array(y)
		return x, y

# 返回可以直接使用的数据集
def load_CIFAR10(root):
	xs = []
	ys = []
	for b in range(1, 6):
		f = os.path.join(root, 'data_batch_%d' % (b))
		x, y = load_cifar_batch(f)
		xs.append(x)
		ys.append(y)
	xtr = np.concatenate(xs)
	ytr = np.concatenate(ys)
	ytr = np.reshape(ytr, (-1, 1))
	print('train:', xtr.shape, ytr.shape)
	x_train = np.concatenate((xtr, ytr), axis=-1)
	del x, y
	xte, yte = load_cifar_batch(os.path.join(root, 'test_batch'))
	yte = np.reshape(yte, (-1, 1))
	x_test = np.concatenate((xte, yte), axis=1)
	return x_train, x_test


def split_feature(arr, paticipantNum=2):
	# 先分割出 feature和label
	data, labels = arr[:, :-1], arr[:, -1].reshape((-1, 1))
	data = np.reshape(data, (-1, 3, 32, 32))
	# arr的shape为：[-1, 3*32*32]
	# 按照通道进行分割数据集，共3072个feature
	if paticipantNum == 2:
		a_channels = 1
		b_channels = 2
		a_feature = data[:, 0:a_channels, :, :]
		b_feature = data[:, a_channels:, :, :]
		a_feature = np.reshape(a_feature, (-1, 1 * 32 * 32))
		b_feature = np.reshape(b_feature, (-1, 2 * 32 * 32))
		b_feature = np.concatenate((b_feature, labels), axis=-1)
		return a_feature, b_feature

	elif paticipantNum == 3:
		# 注意：att最后一个维度是label
		a_channels = 1
		b_channels = 1
		c_channels = 1
		a_feature = data[:, 0:a_channels, :, :]
		b_feature = data[:, a_channels:(a_channels+b_channels), :, :]
		c_feature = data[:, (a_channels+b_channels):, :, :]

		a_feature = np.reshape(a_feature, (-1, 1*32*32))
		b_feature = np.reshape(b_feature, (-1, 1*32*32))
		c_feature = np.reshape(c_feature, (-1, 1*32*32))
		c_feature = np.concatenate((c_feature, labels), axis=-1)
		return a_feature, b_feature, c_feature

	elif paticipantNum == 4:
		# 注意：att最后一个维度是label
		a_channels = 1
		b_channels = 1
		c_channels = 1
		a_feature = data[:, 0:a_channels, :, :]
		b_feature = data[:, a_channels:(a_channels + b_channels), :, :]
		c_feature = data[:, (a_channels + b_channels):, :, :]

		a_feature = np.reshape(a_feature, (-1, 1 * 32 * 32))
		b_feature = np.reshape(b_feature, (-1, 1 * 32 * 32))
		c_feature = np.reshape(c_feature, (-1, 1 * 32 * 32))
		c_feature = np.concatenate((c_feature, labels), axis=-1)
		return a_feature, b_feature, a_feature, c_feature

	elif paticipantNum == 5:
		# 注意：att最后一个维度是label
		a_channels = 1
		b_channels = 1
		c_channels = 1
		a_feature = data[:, 0:a_channels, :, :]
		b_feature = data[:, a_channels:(a_channels + b_channels), :, :]
		c_feature = data[:, (a_channels + b_channels):, :, :]

		a_feature = np.reshape(a_feature, (-1, 1 * 32 * 32))
		b_feature = np.reshape(b_feature, (-1, 1 * 32 * 32))
		c_feature = np.reshape(c_feature, (-1, 1 * 32 * 32))
		c_feature = np.concatenate((c_feature, labels), axis=-1)
		return a_feature, b_feature, a_feature, b_feature, c_feature


if __name__ == '__main__':

	root = '/home/nbic/wangshanshan2023/additiveVFL/data/cifar-10-python'
	train_data, test_data = load_CIFAR10(root)
	print('train-test:', train_data.shape, test_data.shape)
	train_data, val_data = train_test_split(train_data, test_size=0.2)
	# 将数据集保存下来
	np.savez(os.path.join(root, "cifar10.train.npz"), train_data)
	np.savez(os.path.join(root, "cifar10.val.npz"), val_data)

	# 训练集分割为2个参与方
	a, b = split_feature(train_data, paticipantNum=2)
	dirs = os.path.join(root, "twoPaticipants")
	if not os.path.exists(dirs):
		os.makedirs(dirs)
	np.savez(os.path.join(dirs, "train_a.npz"), a)
	np.savez(os.path.join(dirs, "train_b.npz"), b)

	# 训练集分割为3个参与方
	a, b, c = split_feature(train_data, paticipantNum=3)
	dirs = os.path.join(root, "threePaticipants")
	if not os.path.exists(dirs):
		os.makedirs(dirs)
	np.savez(os.path.join(dirs, "train_a.npz"), a)
	np.savez(os.path.join(dirs, "train_b.npz"), b)
	np.savez(os.path.join(dirs, "train_c.npz"), c)

	# 训练集分割为4个参与方
	a, b, c, d = split_feature(train_data, paticipantNum=4)
	dirs = os.path.join(root, "fourPaticipants")
	if not os.path.exists(dirs):
		os.makedirs(dirs)
	np.savez(os.path.join(dirs, "train_a.npz"), a)
	np.savez(os.path.join(dirs, "train_b.npz"), b)
	np.savez(os.path.join(dirs, "train_c.npz"), c)
	np.savez(os.path.join(dirs, "train_d.npz"), d)
	
	# 训练集分割为5个参与方
	a, b, c, d, e = split_feature(train_data, paticipantNum=5)
	dirs = os.path.join(root, "fivePaticipants")
	if not os.path.exists(dirs):
		os.makedirs(dirs)
	np.savez(os.path.join(dirs, "train_a.npz"), a)
	np.savez(os.path.join(dirs, "train_b.npz"), b)
	np.savez(os.path.join(dirs, "train_c.npz"), c)
	np.savez(os.path.join(dirs, "train_d.npz"), d)
	np.savez(os.path.join(dirs, "train_e.npz"), e)

	# 处理验证集
	a, b = split_feature(val_data, paticipantNum=2)
	dirs = os.path.join(root, "twoPaticipants")
	if not os.path.exists(dirs):
		os.makedirs(dirs)
	np.savez(os.path.join(dirs, "val_a.npz"), a)
	np.savez(os.path.join(dirs, "val_b.npz"), b)

	# 训练集分割为3个参与方
	a, b, c = split_feature(val_data, paticipantNum=3)
	dirs = os.path.join(root, "threePaticipants")
	if not os.path.exists(dirs):
		os.makedirs(dirs)
	np.savez(os.path.join(dirs, "val_a.npz"), a)
	np.savez(os.path.join(dirs, "val_b.npz"), b)
	np.savez(os.path.join(dirs, "val_c.npz"), c)

	# 训练集分割为4个参与方
	a, b, c, d = split_feature(val_data, paticipantNum=4)
	dirs = os.path.join(root, "fourPaticipants")
	if not os.path.exists(dirs):
		os.makedirs(dirs)
	np.savez(os.path.join(dirs, "val_a.npz"), a)
	np.savez(os.path.join(dirs, "val_b.npz"), b)
	np.savez(os.path.join(dirs, "val_c.npz"), c)
	np.savez(os.path.join(dirs, "val_d.npz"), d)

	# 训练集分割为5个参与方
	a, b, c, d, e = split_feature(val_data, paticipantNum=5)
	dirs = os.path.join(root, "fivePaticipants")
	if not os.path.exists(dirs):
		os.makedirs(dirs)
	np.savez(os.path.join(dirs, "val_a.npz"), a)
	np.savez(os.path.join(dirs, "val_b.npz"), b)
	np.savez(os.path.join(dirs, "val_c.npz"), c)
	np.savez(os.path.join(dirs, "val_d.npz"), d)
	np.savez(os.path.join(dirs, "val_e.npz"), e)

	# 处理测试集

	np.savez(os.path.join(root, "cifar10.test.npz"), test_data)

	# 测试集分割为2个参与方
	a, b = split_feature(test_data, paticipantNum=2)
	dirs = os.path.join(root, "twoPaticipants")
	if not os.path.exists(dirs):
		os.makedirs(dirs)
	np.savez(os.path.join(dirs, "test_a.npz"), a)
	np.savez(os.path.join(dirs, "test_b.npz"), b)

	# 测试集分割为3个参与方
	a, b, c = split_feature(test_data, paticipantNum=3)
	dirs = os.path.join(root, "threePaticipants")
	if not os.path.exists(dirs):
		os.makedirs(dirs)
	np.savez(os.path.join(dirs, "test_a.npz"), a)
	np.savez(os.path.join(dirs, "test_b.npz"), b)
	np.savez(os.path.join(dirs, "test_c.npz"), c)


	
	# 测试集分割为4个参与方
	a, b, c, d = split_feature(test_data, paticipantNum=4)
	dirs = os.path.join(root,  "fourPaticipants")
	if not os.path.exists(dirs):
		os.makedirs(dirs)
	np.savez(os.path.join(dirs, "test_a.npz"), a)
	np.savez(os.path.join(dirs, "test_b.npz"), b)
	np.savez(os.path.join(dirs, "test_c.npz"), c)
	np.savez(os.path.join(dirs, "test_d.npz"), d)

	# 测试集分割为5个参与方
	a, b, c, d, e = split_feature(test_data, paticipantNum=5)
	dirs = os.path.join(root, "fivePaticipants")
	if not os.path.exists(dirs):
		os.makedirs(dirs)
	np.savez(os.path.join(dirs, "test_a.npz"), a)
	np.savez(os.path.join(dirs, "test_b.npz"), b)
	np.savez(os.path.join(dirs, "test_c.npz"), c)
	np.savez(os.path.join(dirs, "test_d.npz"), d)
	np.savez(os.path.join(dirs, "test_e.npz"), e)



