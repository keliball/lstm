from __future__ import absolute_import
from __future__ import division
from __future__ import print_function






import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
import numpy as np
from numpy import concatenate
from math import sqrt
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib import pyplot
import matplotlib.image as mpimg  # mpimg 用于读取图片
import seaborn as sns
import time

import tensorflow as tf
import os

from PyQt5 import QtCore, QtGui, QtWidgets
from functools import partial
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow


# global temp_epochs
# global temp_batch_size

class Ui_Main_Windows(object):
    def setupUi(self, Main_Windows):
        Main_Windows.setObjectName("Main_Windows")
        Main_Windows.resize(800, 601)
        self.centralwidget = QtWidgets.QWidget(Main_Windows)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(320, 40, 161, 61))
        self.label.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(70, 120, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(70, 180, 201, 31))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.lineEdit = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit.setGeometry(QtCore.QRect(270, 120, 301, 31))
        self.lineEdit.setObjectName("lineEdit")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_2.setGeometry(QtCore.QRect(270, 180, 301, 31))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(420, 280, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(580, 280, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setGeometry(QtCore.QRect(220, 370, 381, 161))
        self.textBrowser.setObjectName("textBrowser")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_3.setGeometry(QtCore.QRect(100, 280, 111, 41))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_4.setGeometry(QtCore.QRect(260, 280, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        Main_Windows.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Main_Windows)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        Main_Windows.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Main_Windows)
        self.statusbar.setObjectName("statusbar")
        Main_Windows.setStatusBar(self.statusbar)

        self.retranslateUi(Main_Windows)
        QtCore.QMetaObject.connectSlotsByName(Main_Windows)

    def retranslateUi(self, Main_Windows):
        _translate = QtCore.QCoreApplication.translate
        Main_Windows.setWindowTitle(_translate("Main_Windows", "ZheLu-LSTM演示实验"))
        self.label.setText(_translate("Main_Windows", "LSTM演示"))
        self.label_2.setText(_translate("Main_Windows", "训练轮数"))
        self.label_3.setText(_translate("Main_Windows", "批处理参数"))
        self.pushButton.setText(_translate("Main_Windows", "训练网络"))
        self.pushButton_2.setText(_translate("Main_Windows", "预测"))
        self.pushButton_3.setText(_translate("Main_Windows", "清洗数据"))
        self.pushButton_4.setText(_translate("Main_Windows", "数据可视化"))


def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')


def read_raw():  # 数据清洗
    print("开始数据清洗，生成pollution.csv数据集")
    dataset = pd.read_csv('raw.csv',
                          parse_dates=[['year', 'month', 'day', 'hour']],
                          index_col=0,
                          date_parser=parse)  # 读取raw.csv

    dataset.drop('No', axis=1, inplace=True)  # 去掉原始数据中的“No”列,inplace会填充nan数据
    # manually specify column names，以下代码手动规定了列的名字
    dataset.columns = [
        '污染指数(PM2.5)', '湿度（水汽压）', '温度（摄氏度）', '气压(帕斯卡)', '风向', '风速(KM/S)',
        '降雪量（毫米）', '降雨量（毫米）'
    ]  # 将列命名为更加清晰的名字
    dataset.index.name = 'date'  # 数据集索引列名字为date
    # mark all NA values with 0 #把所有的NA值改成0，用0填充
    dataset['污染指数(PM2.5)'].fillna(0, inplace=True)  # fillna代表默认参数，设置为0
    # drop the first 24 hours
    dataset = dataset[24:]  # 从第24行到结尾

    # print('******************')
    # print(dataset.dtype)
    # print('******************')

    # summarize first 5 rows
    print("输出前五行数据，展示pollution.csv结果\n\n")
    print(dataset.head(5))  # 输出头五行数据，检查有没有病
    # save to file
    dataset.to_csv('pollution.csv')  # 保存成pollution.csv
    return dataset


def drow_pollution():
    # 现在的数据格式已经更加适合处理，可以简单的对每列进行绘图。下面的代码加载了“pollution.csv”文件，
    # 并对除了类别型特性“风速”的每一列数据分别绘图。
    # zhfont1 = fm.FontProperties(fname='C:\Windows\Fonts\msyh.ttc')

    print("\n\n画出pollution数据集\n\n")
    dataset = pd.read_csv('pollution.csv', header=0,
                          index_col=0)  # 读取生成的pollution数据集
    values = dataset.values
    # specify columns to plot   把列的数据转换成图像
    groups = [0, 1, 2, 3, 5, 6, 7]  # 画出了七列数据
    i = 1
    # plot each column
    plt.figure(num='AI教学试验箱--pollution.csv数据集，横轴代表数据集，纵轴代表对应指数',
               figsize=(10, 10))  # 绘制一个10*10的窗口
    # plt.title(u'AI教学试验箱---LSTM之pollution.csv数据集', fontproperties=zhfont1)
    # plt.subplots(1,1)

    for group in groups:
        plt.subplot(len(groups), 1, i)  # 把一个绘图区域划分为7个len(groups)区域进行绘图
        pyplot.plot(values[:, group])
        pyplot.title(dataset.columns[group],
                     y=0.5,
                     loc='right')  # 每张图的名字，就是列表的名字
        #  fontproperties=zhfont1)
        i += 1
    plt.show()  # 显示画出的图像


# 采用LSTM模型时，第一步需要对数据进行适配处理，其中包括将数据集转化为有监督学习问题和归一化变量（包括输入和输出值）
# 使其能够实现通过前一个时刻（t-1）的污染数据和天气条件预测当前时刻（t）的污染。
"""
Frame a time series as a supervised learning dataset.
Arguments:
	data: Sequence of observations as a list or NumPy array.
	n_in: Number of lag observations as input (X).
	n_out: Number of observations as output (y).
	dropnan: Boolean whether or not to drop rows with NaN values.
Returns:
	Pandas DataFrame of series framed for supervised learning.
"""


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    # convert series to supervised learning
    n_vars = 1 if type(data) is list else data.shape[
        1]  # 如果data已经是个列表，那就不管，否者返回一列一列的形式
    df = pd.DataFrame(data)  # DataFrame是Python中Pandas库中的一种数据结构，它类似excel，是一种二维表。
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)      #输入序列从0~t-1，来作为观测值预测t时刻以后的值
    for i in range(n_in, 0, -1):  # n_in表示起始值，0表示终止值，-1表示步长
        cols.append(df.shift(i))  # shift函数，可以把0转换成NaN，1转换成0，2转换成1
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # print(names)

    # forecast sequence (t, t+1, ... t+n)     #预测t时刻之后的值
    for i in range(0, n_out):  # 0表示起始值，n_out表示终止值
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)  # 当axis = 1的时候，concat就是行对齐，然后将不同列名称的两张表合并
    agg.columns = names  # agg表的列名就是原来每一列的名字
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)  # 不要输出NaN行的数据
    return agg  # 返回生成的有利于监督学习的表


def cs_to_sl():
    # load dataset
    dataset = pd.read_csv('pollution.csv', header=0, index_col=0)  # 读取数据集
    values = dataset.values  # 读取数据集中的值
    # integer encode direction   #整型编码方向
    encoder = LabelEncoder()
    # 将离散型的数据转换成 0 到 n−1 之间的数
    # 这里 n是一个列表的不同取值的个数，可以认为是某个特征的所有不同取值的个数。
    values[:, 4] = encoder.fit_transform(values[:, 4])
    # ensure all data is float
    values = values.astype('float32')  # 把数据转换成32位浮点形式
    # normalize features    特征标准化
    scaler = MinMaxScaler(feature_range=(0, 1))  # 数据归一化，将数据全部变换到0-1之内
    scaled = scaler.fit_transform(values)  # 预处理之后进行转换，计算基本统计量并进行标准化

    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)  # 返回利于监督学习的表
    # drop columns we don't want to predict
    reframed.drop(
        reframed.columns[[9, 10, 11, 12, 13, 14, 15]], axis=1,
        inplace=True)  # drop函数，默认删除行，列需要加axis=1，inplace=true，替换原有的NaN值
    print("输出有利于监督学习序列的前几行：")
    print(reframed.head())  # 输出agg的前几行信息

    # reframed.to_csv('useful framework.csv')
    # print('save successfully')

    return reframed, scaler


def train_test(reframed):
    # split into train and test sets
    values = reframed.values
    n_train_hours = 365 * 24
    train = values[:n_train_hours, :]
    print("\n\n数据转换之后的训练集")
    print(train)

    test = values[n_train_hours:, :]
    print("\n\n数据转换之后的测试集\n\n")
    print(test)
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print("\n\n训练集的形状")
    print(train_X.shape, train_y.shape)
    print("测试集的形状")
    print(test_X.shape, test_y.shape)

    return train, test, train_X, train_y, test_X, test_y

class DropoutRNNCellMixin(object):
  def __init__(self, *args, **kwargs):
    self._dropout_mask = None
    self._recurrent_dropout_mask = None
    self._eager_dropout_mask = None
    self._eager_recurrent_dropout_mask = None
    super(DropoutRNNCellMixin, self).__init__(*args, **kwargs)

  def reset_dropout_mask(self):
    self._dropout_mask = None
    self._eager_dropout_mask = None

  def reset_recurrent_dropout_mask(self):
    self._recurrent_dropout_mask = None
    self._eager_recurrent_dropout_mask = None

  def get_dropout_mask_for_cell(self, inputs, training, count=1):
    if self.dropout == 0:
      return None
    if (not context.executing_eagerly() and self._dropout_mask is None
        or context.executing_eagerly() and self._eager_dropout_mask is None):
      # Generate new mask and cache it based on context.
      dp_mask = _generate_dropout_mask(
          array_ops.ones_like(inputs),
          self.dropout,
          training=training,
          count=count)
      if context.executing_eagerly():
        self._eager_dropout_mask = dp_mask
      else:
        self._dropout_mask = dp_mask
    else:
      # Reuse the existing mask.
      dp_mask = (self._eager_dropout_mask
                 if context.executing_eagerly() else self._dropout_mask)
    return dp_mask

  def get_recurrent_dropout_mask_for_cell(self, inputs, training, count=1):
    if self.recurrent_dropout == 0:
      return None
    if (not context.executing_eagerly() and self._recurrent_dropout_mask is None
        or context.executing_eagerly()
        and self._eager_recurrent_dropout_mask is None):
      # Generate new mask and cache it based on context.
      rec_dp_mask = _generate_dropout_mask(
          array_ops.ones_like(inputs),
          self.recurrent_dropout,
          training=training,
          count=count)
      if context.executing_eagerly():
        self._eager_recurrent_dropout_mask = rec_dp_mask
      else:
        self._recurrent_dropout_mask = rec_dp_mask
    else:
      # Reuse the existing mask.
      rec_dp_mask = (self._eager_recurrent_dropout_mask
                     if context.executing_eagerly()
                     else self._recurrent_dropout_mask)
    return rec_dp_mask




