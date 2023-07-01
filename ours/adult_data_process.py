#! usr/bin/env python 39
# -*- coding: utf-8 -*- 
# @Time  : 2023/3/29 20:20
# @Author: sharonswang   
# @File  : adult_data_process.py

import os
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

attr_classes = {
    "workclass": ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay',
                  'Never-worked', 'Retired'],
    "education": ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th',
                  '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
    "marital-status": ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed',
                       'Married-spouse-absent', 'Married-AF-spouse'],
    "occupation": ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty',
                   'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
                   'Priv-house-serv', 'Protective-serv', 'Armed-Forces', 'Retired', 'Student', 'None'],
    "relationship": ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
    "race": ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
    "sex": ['Female', 'Male'],
    "native-country": ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
                       'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran',
                       'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal',
                       'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia',
                       'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador',
                       'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'],
    "label": ['<=50K', '>50K']
    }

def get_mean_std(df):
    res = {}
    for col in ["age", "capital-gain", "capital-loss", "hours-per-week"]:
        mean = df[col].mean()
        std = df[col].std()
        res[col] = {'mean': mean, 'std': std}
    return res

def int_to_one_hot(val, total):
    res = [0] * total
    if val >= 0:
        res[val] = 1
    return res

def df_to_arr(df):
    names = df.columns.tolist()
    print('names:', names)
    rows = len(df)

    arr = []
    for i in range(rows):
        arr_row = []
        for col in names:
            if col in attr_classes:
                raw_val = df.iloc[i][col].strip()
                if raw_val == "?":
                    val = -1
                else:
                    val = attr_classes[col].index(raw_val)
                if col == "label":
                    arr_row.append(val)
                else:
                    arr_row.extend(int_to_one_hot(val, len(attr_classes[col])))
            else:
                val = df.iloc[i][col]
                arr_row.append(val)
        arr.append(arr_row)
    res = np.array(arr, dtype=np.float32)
    return res
def isint(x):
    x = str(x)
    x = float(x)
    return x
def convert_csv_to_arr(df, mean_std):
    # 移除无用的列
    df.drop(columns=["fnlwgt", "education-num"], inplace=True)
    # 填充空值
    df.loc[df.workclass == "Never-worked", ["occupation"]] = 'None'
    #df['age'] = df.age.map(isint)
    df.loc[(df.age < 24) & (df.occupation == "?"), ["workclass", "occupation"]] = ["Never-worked", "Student"]
    df.loc[(df.age > 60) & (df.occupation == "?"), ["workclass", "occupation"]] = ["Retired", "Retired"]

    # 归一化连续性数值
    for col in mean_std:
        mean = mean_std[col]['mean']
        std = mean_std[col]['std']
        df[col] = (df[col]-mean)/std

    arr = df_to_arr(df)
    # shuffle 数据集
    from sklearn.utils import shuffle
    arr = shuffle(arr)
    return arr

def split_feature(arr, paticipantNum=2):

    if paticipantNum == 2:

        a_feature_size = 50
        b_feature_size = 57

        a_feature = arr[:, :a_feature_size]
        b_feature = arr[:, a_feature_size:]
        return a_feature, b_feature

    elif paticipantNum == 3:
        # 注意：att最后一个维度是label
        a_feature_size = 30
        b_feature_size = 40
        c_feature_with_label_size = 38

        a_feature = arr[:, :a_feature_size]
        b_feature = arr[:, a_feature_size:(a_feature_size+b_feature_size)]
        c_feature_with_label = arr[:, (a_feature_size+b_feature_size):]
        return a_feature, b_feature, c_feature_with_label

    elif paticipantNum == 4:
        # 注意：att最后一个维度是label
        a_feature_size = 20
        b_feature_size = 30
        c_feature_size = 30
        d_feature_with_label_size = 28

        a_feature = arr[:, :a_feature_size]
        b_feature = arr[:, a_feature_size:(a_feature_size+b_feature_size)]
        c_feature = arr[:, (a_feature_size+b_feature_size): (a_feature_size+b_feature_size+c_feature_size)]
        d_feature_with_label = arr[:, (a_feature_size+b_feature_size+c_feature_size):]
        return a_feature, b_feature, c_feature, d_feature_with_label

    elif paticipantNum == 5:
        # 注意：att最后一个维度是label
        a_feature_size = 20
        b_feature_size = 20
        c_feature_size = 20
        d_feature_size = 20
        e_feature_with_label_size = 28

        a_feature = arr[:, :a_feature_size]
        b_feature = arr[:, a_feature_size:(a_feature_size + b_feature_size)]
        c_feature = arr[:, (a_feature_size + b_feature_size): (a_feature_size + b_feature_size + c_feature_size)]
        d_feature = arr[:, (a_feature_size + b_feature_size + c_feature_size):(a_feature_size + b_feature_size + c_feature_size+d_feature_size)]
        e_feature_with_label = arr[:, (a_feature_size + b_feature_size + c_feature_size+d_feature_size):]
        return a_feature, b_feature, c_feature, d_feature, e_feature_with_label

if __name__ == '__main__':
    if not os.path.exists('../data/adult'):
        os.makedirs('../data/adult/a', exist_ok=True)
        os.makedirs('../data/adult/b', exist_ok=True)
        os.makedirs('../data/adult/c', exist_ok=True)

    names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
             "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
             "label"]
    df = pd.read_csv('../data/adult/adult.data', sep=',', names=names)
    print(df.head(5))

    mean_std = get_mean_std(df)
    arr = convert_csv_to_arr(df, mean_std)
    # 将训练集划分为训练集和测试集
    train_arr, val_arr = train_test_split(arr, test_size=0.2)
    print(train_arr.shape, val_arr.shape)
    np.savez("../data/adult/adult.train.npz", train_arr)
    np.savez("../data/adult/adult.val.npz", val_arr)

    # 训练集分割为2个参与方
    a, b = split_feature(train_arr, paticipantNum=2)
    dirs = "../data/adult/twoPaticipants"
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    np.savez(os.path.join(dirs, "train_a.npz"), a)
    np.savez(os.path.join(dirs, "train_b.npz"), b)

    # 训练集分割为3个参与方
    a, b, c = split_feature(train_arr, paticipantNum=3)
    dirs = "../data/adult/threePaticipants"
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    np.savez(os.path.join(dirs, "train_a.npz"), a)
    np.savez(os.path.join(dirs, "train_b.npz"), b)
    np.savez(os.path.join(dirs, "train_c.npz"), c)

    # 训练集分割为4个参与方
    a, b, c, d = split_feature(train_arr, paticipantNum=4)
    dirs = "../data/adult/fourPaticipants"
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    np.savez(os.path.join(dirs, "train_a.npz"), a)
    np.savez(os.path.join(dirs, "train_b.npz"), b)
    np.savez(os.path.join(dirs, "train_c.npz"), c)
    np.savez(os.path.join(dirs, "train_d.npz"), d)

    # 训练集分割为5个参与方
    a, b, c, d, e = split_feature(train_arr, paticipantNum=5)
    dirs = "../data/adult/fivePaticipants"
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    np.savez(os.path.join(dirs, "train_a.npz"), a)
    np.savez(os.path.join(dirs, "train_b.npz"), b)
    np.savez(os.path.join(dirs, "train_c.npz"), c)
    np.savez(os.path.join(dirs, "train_d.npz"), d)
    np.savez(os.path.join(dirs, "train_e.npz"), e)

    # 处理验证集， 验证集划分为2个参与方
    a, b = split_feature(val_arr, paticipantNum=2)
    dirs = "../data/adult/twoPaticipants"
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    np.savez(os.path.join(dirs, "val_a.npz"), a)
    np.savez(os.path.join(dirs, "val_b.npz"), b)

    # 验证集划分为3个参与方
    a, b, c = split_feature(val_arr, paticipantNum=3)
    dirs = "../data/adult/threePaticipants"
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    np.savez(os.path.join(dirs, "val_a.npz"), a)
    np.savez(os.path.join(dirs, "val_b.npz"), b)
    np.savez(os.path.join(dirs, "val_c.npz"), c)

    # 验证集分割为4个参与方
    a, b, c, d = split_feature(val_arr, paticipantNum=4)
    dirs = "../data/adult/fourPaticipants"
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    np.savez(os.path.join(dirs, "val_a.npz"), a)
    np.savez(os.path.join(dirs, "val_b.npz"), b)
    np.savez(os.path.join(dirs, "val_c.npz"), c)
    np.savez(os.path.join(dirs, "val_d.npz"), d)

    # 验证集分割为5个参与方
    a, b, c, d, e = split_feature(val_arr, paticipantNum=5)
    dirs = "../data/adult/fivePaticipants"
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    np.savez(os.path.join(dirs, "val_a.npz"), a)
    np.savez(os.path.join(dirs, "val_b.npz"), b)
    np.savez(os.path.join(dirs, "val_c.npz"), c)
    np.savez(os.path.join(dirs, "val_d.npz"), d)
    np.savez(os.path.join(dirs, "val_e.npz"), e)

    # 处理测试集
    test_df = pd.read_csv('../data/adult/adult.test', names=names)
    arr = convert_csv_to_arr(test_df, mean_std)
    np.savez("../data/adult/adult.test.npz", arr)

    # 测试集分割为2个参与方
    a, b = split_feature(arr, paticipantNum=2)
    dirs = "../data/adult/twoPaticipants"
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    np.savez(os.path.join(dirs, "test_a.npz"), a)
    np.savez(os.path.join(dirs, "test_b.npz"), b)

    # 测试集分割为3个参与方
    a, b, c = split_feature(arr, paticipantNum=3)
    dirs = "../data/adult/threePaticipants"
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    np.savez(os.path.join(dirs, "test_a.npz"), a)
    np.savez(os.path.join(dirs, "test_b.npz"), b)
    np.savez(os.path.join(dirs, "test_c.npz"), c)

    # 测试集分割为4个参与方
    a, b, c, d = split_feature(arr, paticipantNum=4)
    dirs = "../data/adult/fourPaticipants"
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    np.savez(os.path.join(dirs, "test_a.npz"), a)
    np.savez(os.path.join(dirs, "test_b.npz"), b)
    np.savez(os.path.join(dirs, "test_c.npz"), c)
    np.savez(os.path.join(dirs, "test_d.npz"), d)

    # 测试集分割为5个参与方
    a, b, c, d, e = split_feature(arr, paticipantNum=5)
    dirs = "../data/adult/fivePaticipants"
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    np.savez(os.path.join(dirs, "test_a.npz"), a)
    np.savez(os.path.join(dirs, "test_b.npz"), b)
    np.savez(os.path.join(dirs, "test_c.npz"), c)
    np.savez(os.path.join(dirs, "test_d.npz"), d)
    np.savez(os.path.join(dirs, "test_e.npz"), e)

