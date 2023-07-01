# -*- coding: utf-8 -*-
# @Time  : 2023/4/20 16:12
# @Author: sharonswang   
# @File  : kchouse_data_process.py

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class SongsPopularityTransformer():
    def __init__(self):
        pass
    def transform(self, db):
        db['loudness'] = db['loudness'].abs() #converting values to positive
        db['loudness'] = (db['loudness'] - np.min(db['loudness']))/(np.max(db['loudness']) - np.min(db['loudness']))
        #scaling values range to be between 0 and 1
        db = db[['song_popularity', 'acousticness','energy', 'loudness', 'instrumentalness', 'liveness']]
        return db

def df_norm(df, *cols):
	df_n = df.copy()
	for col in cols:
		ma = df_n[col].max()
		mi = df_n[col].min()
		df_n[col] = (df_n[col]-mi)/(ma-mi)
	return df_n

def load_song_price_data(data_root):
	data = pd.read_csv(data_root)
	print(data.sample())
	# drop id和date这两列，因为对任务没有关系
	data.drop(['song_name'], axis='columns', inplace=True)
	print(data.isnull().sum())
	data = df_norm(data, ['song_duration_ms', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'audio_valence'])
	data = np.array(data)

	# 将price这一列（即label对应的列放在最后）
	labels = data[:, 0:1]
	features = data[:, 1:]

	data = np.concatenate((features, labels), axis=1)
	print(data.shape)   # (18835, 14)

	# 分割训练集和测试集
	train_data, test_data = train_test_split(data, test_size=0.2)
	# 再次分割出来验证集
	train_data, val_data = train_test_split(train_data, test_size=0.2)
	return train_data, val_data, test_data

def split_feature(arr, paticipantNum=2):
	# arr的shape为：[-1, 3*32*32]
	# 按照通道进行分割数据集，共3072个feature
	if paticipantNum == 2:
		a_feature_size = 7
		b_feature_size = 7
		a_feature = arr[:, :a_feature_size]
		b_feature = arr[:, a_feature_size:]
		print(a_feature.shape, b_feature.shape)  # (4323,9) (4323,10)
		return a_feature, b_feature

	elif paticipantNum == 3:
		# 注意：att最后一个维度是label
		a_feature_size = 4
		b_feature_size = 4
		c_feature_with_label_size = 6
		a_feature = arr[:, :a_feature_size]
		b_feature = arr[:, a_feature_size:(a_feature_size+b_feature_size)]
		c_feature = arr[:, (a_feature_size+b_feature_size):]
		return a_feature, b_feature, c_feature

	elif paticipantNum == 4:
		# 注意：att最后一个维度是label
		a_feature_size = 3
		b_feature_size = 3
		c_feature_size = 3
		d_feature_with_label_size = 5

		a_feature = arr[:, :a_feature_size]
		b_feature = arr[:, a_feature_size:(a_feature_size+b_feature_size)]
		c_feature = arr[:, (a_feature_size+b_feature_size): (a_feature_size+b_feature_size+c_feature_size)]
		d_feature = arr[:, (a_feature_size+b_feature_size+c_feature_size):]
		return a_feature, b_feature, c_feature, d_feature

	elif paticipantNum == 5:
		# 注意：att最后一个维度是label
		a_feature_size = 2
		b_feature_size = 2
		c_feature_size = 2
		d_feature_size = 2
		e_feature = 6

		a_feature = arr[:, :a_feature_size]
		b_feature = arr[:, a_feature_size:(a_feature_size + b_feature_size)]
		c_feature = arr[:, (a_feature_size + b_feature_size): (a_feature_size + b_feature_size + c_feature_size)]
		d_feature = arr[:, (a_feature_size + b_feature_size + c_feature_size):(a_feature_size + b_feature_size + c_feature_size+d_feature_size)]
		e_feature = arr[:, (a_feature_size + b_feature_size + c_feature_size+d_feature_size):]
		return a_feature, b_feature, c_feature, d_feature, e_feature


if __name__ == '__main__':
    # 加载数据集
	root = '/home/nbic/wangshanshan2023/additiveVFL/data/song'
	train_data, val_data, test_data = load_song_price_data(os.path.join(root, 'song_data.csv'))

	# 将数据集保存下来
	np.savez(os.path.join(root, "song.train.npz"), train_data)
	np.savez(os.path.join(root, "song.val.npz"), val_data)

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

	# 处理验证集， 验证集划分为2个参与方
	a, b = split_feature(val_data, paticipantNum=2)
	dirs = os.path.join(root, "twoPaticipants")
	if not os.path.exists(dirs):
		os.makedirs(dirs)
	np.savez(os.path.join(dirs, "val_a.npz"), a)
	np.savez(os.path.join(dirs, "val_b.npz"), b)

	# 验证集划分为3个参与方
	a, b, c = split_feature(val_data, paticipantNum=3)
	dirs = os.path.join(root, "threePaticipants")
	if not os.path.exists(dirs):
		os.makedirs(dirs)
	np.savez(os.path.join(dirs, "val_a.npz"), a)
	np.savez(os.path.join(dirs, "val_b.npz"), b)
	np.savez(os.path.join(dirs, "val_c.npz"), c)

	# 验证集分割为4个参与方
	a, b, c, d = split_feature(val_data, paticipantNum=4)
	dirs = os.path.join(root, "fourPaticipants")
	if not os.path.exists(dirs):
		os.makedirs(dirs)
	np.savez(os.path.join(dirs, "val_a.npz"), a)
	np.savez(os.path.join(dirs, "val_b.npz"), b)
	np.savez(os.path.join(dirs, "val_c.npz"), c)
	np.savez(os.path.join(dirs, "val_d.npz"), d)

	# 验证集分割为5个参与方
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

	np.savez(os.path.join(root, "song.test.npz"), test_data)

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
	dirs = os.path.join(root, "fourPaticipants")
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

