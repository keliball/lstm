from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.keras.layers import RNN
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.training.tracking import data_structures
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import keras_export
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM
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
from Instar.ex2_9.Instar_Public_ex2_9 import drow_pollution
from Instar.ex2_9.Instar_Public_ex2_9 import series_to_supervised
from Instar.ex2_9.Instar_Public_ex2_9 import cs_to_sl
from Instar.ex2_9.Instar_Public_ex2_9 import train_test
from Instar.ex2_9.Instar_Public_ex2_9 import read_raw
from Instar.ex2_9.Instar_Public_ex2_9 import Ui_Main_Windows
from Instar.ex2_9.Instar_Public_ex2_9 import DropoutRNNCellMixin

@keras_export(v1=['keras.layers.LSTMCell'])
class LSTMCell(DropoutRNNCellMixin, Layer):
  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               **kwargs):
    super(LSTMCell, self).__init__(**kwargs)
    self.units = units
    self.activation = activations.get(activation)
    self.recurrent_activation = activations.get(recurrent_activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.unit_forget_bias = unit_forget_bias

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
    self.implementation = implementation
    # tuple(_ListWrapper) was silently dropping list content in at least 2.7.10,
    # and fixed after 2.7.16. Converting the state_size to wrapper around
    # NoDependency(), so that the base_layer.__setattr__ will not convert it to
    # ListWrapper. Down the stream, self.states will be a list since it is
    # generated from nest.map_structure with list, and tuple(list) will work
    # properly.
    self.state_size = data_structures.NoDependency([self.units, self.units])
    self.output_size = self.units

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    input_dim = input_shape[-1]
    self.kernel = self.add_weight(
        shape=(input_dim, self.units * 4),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)
    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units * 4),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)

    if self.use_bias:
      if self.unit_forget_bias:

        def bias_initializer(_, *args, **kwargs):
          return K.concatenate([
              self.bias_initializer((self.units,), *args, **kwargs),
              initializers.Ones()((self.units,), *args, **kwargs),
              self.bias_initializer((self.units * 2,), *args, **kwargs),
          ])
      else:
        bias_initializer = self.bias_initializer
      self.bias = self.add_weight(
          shape=(self.units * 4,),
          name='bias',
          initializer=bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint)
    else:
      self.bias = None
    self.built = True

  def _compute_carry_and_output(self, x, h_tm1, c_tm1):
    """Computes carry and output using split kernels."""
    x_i, x_f, x_c, x_o = x
    h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1
    i = self.recurrent_activation(
        x_i + K.dot(h_tm1_i, self.recurrent_kernel[:, :self.units]))
    f = self.recurrent_activation(x_f + K.dot(
        h_tm1_f, self.recurrent_kernel[:, self.units:self.units * 2]))
    c = f * c_tm1 + i * self.activation(x_c + K.dot(
        h_tm1_c, self.recurrent_kernel[:, self.units * 2:self.units * 3]))
    o = self.recurrent_activation(
        x_o + K.dot(h_tm1_o, self.recurrent_kernel[:, self.units * 3:]))
    return c, o

  def _compute_carry_and_output_fused(self, z, c_tm1):
    """Computes carry and output using fused kernels."""
    z0, z1, z2, z3 = z
    i = self.recurrent_activation(z0)
    f = self.recurrent_activation(z1)
    c = f * c_tm1 + i * self.activation(z2)
    o = self.recurrent_activation(z3)
    return c, o

  def call(self, inputs, states, training=None):
    h_tm1 = states[0]  # previous memory state
    c_tm1 = states[1]  # previous carry state

    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
    rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
        h_tm1, training, count=4)

    if self.implementation == 1:
      if 0 < self.dropout < 1.:
        inputs_i = inputs * dp_mask[0]
        inputs_f = inputs * dp_mask[1]
        inputs_c = inputs * dp_mask[2]
        inputs_o = inputs * dp_mask[3]
      else:
        inputs_i = inputs
        inputs_f = inputs
        inputs_c = inputs
        inputs_o = inputs
      k_i, k_f, k_c, k_o = array_ops.split(
          self.kernel, num_or_size_splits=4, axis=1)
      x_i = K.dot(inputs_i, k_i)
      x_f = K.dot(inputs_f, k_f)
      x_c = K.dot(inputs_c, k_c)
      x_o = K.dot(inputs_o, k_o)
      if self.use_bias:
        b_i, b_f, b_c, b_o = array_ops.split(
            self.bias, num_or_size_splits=4, axis=0)
        x_i = K.bias_add(x_i, b_i)
        x_f = K.bias_add(x_f, b_f)
        x_c = K.bias_add(x_c, b_c)
        x_o = K.bias_add(x_o, b_o)

      if 0 < self.recurrent_dropout < 1.:
        h_tm1_i = h_tm1 * rec_dp_mask[0]
        h_tm1_f = h_tm1 * rec_dp_mask[1]
        h_tm1_c = h_tm1 * rec_dp_mask[2]
        h_tm1_o = h_tm1 * rec_dp_mask[3]
      else:
        h_tm1_i = h_tm1
        h_tm1_f = h_tm1
        h_tm1_c = h_tm1
        h_tm1_o = h_tm1
      x = (x_i, x_f, x_c, x_o)
      h_tm1 = (h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o)
      c, o = self._compute_carry_and_output(x, h_tm1, c_tm1)
    else:
      if 0. < self.dropout < 1.:
        inputs *= dp_mask[0]
      z = K.dot(inputs, self.kernel)
      if 0. < self.recurrent_dropout < 1.:
        h_tm1 *= rec_dp_mask[0]
      z += K.dot(h_tm1, self.recurrent_kernel)
      if self.use_bias:
        z = K.bias_add(z, self.bias)
      z = array_ops.split(z, num_or_size_splits=4, axis=1)
      c, o = self._compute_carry_and_output_fused(z, c_tm1)



    # TODO3:计算h
    # 说明：计算h
    # 提示：1.请根据文档中的原理部分，在o和c已知的情况下，计算h.
    #       2.activation( )为tanh激活函数.
    #
    # ====================================

    return h, [h, c]

  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'unit_forget_bias':
            self.unit_forget_bias,
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
        'implementation':
            self.implementation
    }
    base_config = super(LSTMCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return list(_generate_zero_filled_state_for_cell(
        self, inputs, batch_size, dtype))


@keras_export('keras.experimental.PeepholeLSTMCell')
class PeepholeLSTMCell(LSTMCell):
  def build(self, input_shape):
    super(PeepholeLSTMCell, self).build(input_shape)
    # The following are the weight matrices for the peephole connections. These
    # are multiplied with the previous internal state during the computation of
    # carry and output.
    self.input_gate_peephole_weights = self.add_weight(
        shape=(self.units,),
        name='input_gate_peephole_weights',
        initializer=self.kernel_initializer)
    self.forget_gate_peephole_weights = self.add_weight(
        shape=(self.units,),
        name='forget_gate_peephole_weights',
        initializer=self.kernel_initializer)
    self.output_gate_peephole_weights = self.add_weight(
        shape=(self.units,),
        name='output_gate_peephole_weights',
        initializer=self.kernel_initializer)

  def _compute_carry_and_output(self, x, h_tm1, c_tm1):
    x_i, x_f, x_c, x_o = x
    h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1

    # TODO2:计算i，f，c，o
    # 说明：计算i，f，c，o
    # 提示：1.请根据文档中的原理部分，计算i，f，c，o.
    #       2.文档中的self.recurrent_kernel_i等并不能直接使用，请结合5.2.7首先计算self.recurrent_kernel_i等四个参量.
    #1212125454545
    # ====================================

    return c, o

  def _compute_carry_and_output_fused(self, z, c_tm1):
    z0, z1, z2, z3 = z
    i = self.recurrent_activation(z0 +
                                  self.input_gate_peephole_weights * c_tm1)
    f = self.recurrent_activation(z1 +
                                  self.forget_gate_peephole_weights * c_tm1)
    c = f * c_tm1 + i * self.activation(z2)
    o = self.recurrent_activation(z3 + self.output_gate_peephole_weights * c)
    return c, o

@keras_export(v1=['keras.layers.LSTM'])
class LSTM(RNN):
  def __init__(self,
               units,
               activation='tanh',
               recurrent_activation='hard_sigmoid',
               use_bias=True,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               unit_forget_bias=True,
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               implementation=1,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               **kwargs):
    if implementation == 0:
      logging.warning('`implementation=0` has been deprecated, '
                      'and now defaults to `implementation=1`.'
                      'Please update your layer call.')
    cell = LSTMCell(
        units,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        recurrent_initializer=recurrent_initializer,
        unit_forget_bias=unit_forget_bias,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        recurrent_regularizer=recurrent_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        recurrent_constraint=recurrent_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation)
    super(LSTM, self).__init__(
        cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        **kwargs)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.input_spec = [InputSpec(ndim=3)]

  def call(self, inputs, mask=None, training=None, initial_state=None):
    self.cell.reset_dropout_mask()
    self.cell.reset_recurrent_dropout_mask()
    return super(LSTM, self).call(
        inputs, mask=mask, training=training, initial_state=initial_state)

  @property
  def units(self):
    return self.cell.units

  @property
  def activation(self):
    return self.cell.activation

  @property
  def recurrent_activation(self):
    return self.cell.recurrent_activation

  @property
  def use_bias(self):
    return self.cell.use_bias

  @property
  def kernel_initializer(self):
    return self.cell.kernel_initializer

  @property
  def recurrent_initializer(self):
    return self.cell.recurrent_initializer

  @property
  def bias_initializer(self):
    return self.cell.bias_initializer

  @property
  def unit_forget_bias(self):
    return self.cell.unit_forget_bias

  @property
  def kernel_regularizer(self):
    return self.cell.kernel_regularizer

  @property
  def recurrent_regularizer(self):
    return self.cell.recurrent_regularizer

  @property
  def bias_regularizer(self):
    return self.cell.bias_regularizer

  @property
  def kernel_constraint(self):
    return self.cell.kernel_constraint

  @property
  def recurrent_constraint(self):
    return self.cell.recurrent_constraint

  @property
  def bias_constraint(self):
    return self.cell.bias_constraint

  @property
  def dropout(self):
    return self.cell.dropout

  @property
  def recurrent_dropout(self):
    return self.cell.recurrent_dropout

  @property
  def implementation(self):
    return self.cell.implementation

  def get_config(self):
    config = {
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'unit_forget_bias':
            self.unit_forget_bias,
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
        'implementation':
            self.implementation
    }
    base_config = super(LSTM, self).get_config()
    del base_config['cell']
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    if 'implementation' in config and config['implementation'] == 0:
      config['implementation'] = 1
    return cls(**config)


def _generate_dropout_mask(ones, rate, training=None, count=1):
  def dropped_inputs():
    return K.dropout(ones, rate)

  if count > 1:
    return [
        K.in_train_phase(dropped_inputs, ones, training=training)
        for _ in range(count)
    ]
  return K.in_train_phase(dropped_inputs, ones, training=training)


def _standardize_args(inputs, initial_state, constants, num_constants):
  if isinstance(inputs, list):
    assert initial_state is None and constants is None
    if num_constants is not None:
      constants = inputs[-num_constants:]
      inputs = inputs[:-num_constants]
    if len(inputs) > 1:
      initial_state = inputs[1:]
      inputs = inputs[:1]

    if len(inputs) > 1:
      inputs = tuple(inputs)
    else:
      inputs = inputs[0]

  def to_list_or_none(x):
    if x is None or isinstance(x, list):
      return x
    if isinstance(x, tuple):
      return list(x)
    return [x]

  initial_state = to_list_or_none(initial_state)
  constants = to_list_or_none(constants)

  return inputs, initial_state, constants


def _is_multiple_state(state_size):
  """Check whether the state_size contains multiple states."""
  return (hasattr(state_size, '__len__') and
          not isinstance(state_size, tensor_shape.TensorShape))


def _generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
  if inputs is not None:
    batch_size = array_ops.shape(inputs)[0]
    dtype = inputs.dtype
  return _generate_zero_filled_state(batch_size, cell.state_size, dtype)


def _generate_zero_filled_state(batch_size_tensor, state_size, dtype):
  """Generate a zero filled tensor with shape [batch_size, state_size]."""
  if batch_size_tensor is None or dtype is None:
    raise ValueError(
        'batch_size and dtype cannot be None while constructing initial state: '
        'batch_size={}, dtype={}'.format(batch_size_tensor, dtype))

  def create_zeros(unnested_state_size):
    flat_dims = tensor_shape.as_shape(unnested_state_size).as_list()
    init_state_size = [batch_size_tensor] + flat_dims
    return array_ops.zeros(init_state_size, dtype=dtype)

  if nest.is_sequence(state_size):
    return nest.map_structure(create_zeros, state_size)
  else:
    return create_zeros(state_size)

def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')


# 前期数据可视化分析


def preview():
    # TODO1:载入数据
    # 说明：请完成数据集raw.csv读取载入的工作
    # 提示：1.请调用pandas模块中的read_csv()方法，并将载入的数据集存入data变量中
    #      2.read_csv()方法所需参数说明：
    #      parse_dates参数用于重构列，这里可使用这个参数对日期进行重组。
    #      为了接下来的工作，请将这个参数统一设置为列表[['year', 'month', 'day', 'hour']]，这样不必修改parse()函数。
    #      index_col参数用于设置将某一列作为index.
    #      date_parser参数可将指定将输入的字符串转换为可变的时间数据，该参数需要传入一个函数，该函数已经被定义为parse().
    #
    # ====================================


    print(data.head())

    # data.head()
    print('previewStart')  # 输出几行数据查看情况
    print(data['pm2.5'].describe())  # 查看PM2.5目标类型和分布、有无异常值

    # sns.distplot(x)  #查看PM2.5分布的散点图
    # plt.show()

    sns.jointplot(x='TEMP', y='pm2.5', data=data)  # 查看PM2.5和温度的分布图
    plt.show()
    sns.jointplot(x='DEWP', y='pm2.5', data=data)  # 查看PM2.5和露点的分布图
    plt.show()
    sns.jointplot(x='PRES', y='pm2.5', data=data)  # 查看PM2.5和大气压的分布图
    plt.show()
    # 可以使用热力图表示变量相关性
    plt.rcParams['figure.figsize'] = (15, 10)  # 计算相关系数
    corrmatrix = data.corr()
    sns.heatmap(corrmatrix,
                square=True,
                vmax=1,
                vmin=-1,
                center=0.0,
                cmap='coolwarm')

    plt.show()

def fit_network(train_X, train_y, test_X, test_y, scaler, train_epochs, train_batch_size):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # train_epochs = input("\n\n请输入训练轮数（0-150）以内:")
    # train_batch_size = input("请输入批处理参数(24的倍数):")
    print("开始", int(train_epochs),
          "轮的训练，epoch代表第几轮，loss代表训练集误差，val_loss代表验证集误差\n\n")

    # 网络结构（核心代码）
    ''' LSTM()方法的核心参数主要有
        units：输出维度
        input_dim：输入维度，当使用该层为模型首层时，应指定该值（或等价的指定input_shape)
        return_sequences：布尔值，默认False，控制返回类型。若为True则返回整个序列，否则仅返回输出序列的最后一个输出
        input_length：当输入序列的长度固定时，该参数为输入序列的长度。当需要在该层后连接Flatten层，然后又要连接Dense层时，
        需要指定该参数，否则全连接的输出无法计算出来。
        Dense()方法的某个方法重载只有一个参数，即神经元数
        compile()方法完整地原型是
        compile(optimizer,
                loss=None,
                metrics=None,
                loss_weights=None,
                sample_weight_mode=None,
                weighted_metrics=None,
                target_tensors=None,
                **kwargs)
        其中，optimiizer为优化器，loss为损失函数'''

    model = Sequential()
    # TODO4:网络结构
    # 说明：请完成网络结构设计
    # 提示：1.请认真阅读上述函数说明，结合以下提示完成。
    #       2.使用model.add()方法对网络结构进行构建。
    #       3.本例中的网络，LSTM输出维度可以设置为50，后需要连接1个神经元的全连接层。
    #       4.调用LSTM方法时，请注意利用input_shape参数对输入数据的维度进行控制。
    #       5.损失函数为MSE函数，优化器请采用Adagrad优化器。
    #
    # ====================================

    # fit network
    '''model.fit()方法用于进行训练，这个方法的完整重载是

    fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, 
    validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, 
    sample_weight=None, initial_epoch=0, steps_per_epoch=None,
    validation_steps=None, validation_freq=1)

    一般地，我们需要以下几个重要参数
    model.fit( 训练集的输入特征，
                 训练集的标签，  
                 batch_size,  #每一个batch的大小
                 epochs,   #迭代次数
                 validation_data = (测试集的输入特征，测试集的标签），
                 validation_split = 从测试集中划分多少比例给训练集，
                 validation_freq = 测试的epoch间隔数
                 verbose = 日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
                 shuffle = 一般为布尔值，表示是否在训练过程中随机打乱输入样本的顺序。）'''

    # TODO5:网络训练
    # 说明：请完成网络训练，并将该过程储存在history中
    # 提示：1.请认真阅读上述函数说明，结合以下提示完成。
    #       2.本例中，需要的参数有
    #                  训练集的输入特征x，
    #                  训练集的标签y，
    #                  batch_size,
    #                  epochs,
    #                  validation_data，
    #                  verbose，这里为了方便演示，请设置为2
    #                  shuffle.考察设置该参数的原因
    #
    # ====================================


    # plot history
    print("画出训练集和验证集之间的误差")
    pyplot.figure(num='预测集合和测试集合的误差', figsize=(10, 10))
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    # make a prediction
    model.save('model_weather')
    print("\n\n模型保存在当前目录，为model_weather\n\n")
    network_struct_picture = mpimg.imread(
        'network.png')  # 读取和代码处于同一目录下的 lena.png
    network_struct_picture.shape  # (512, 512, 3)
    pyplot.imshow(network_struct_picture)  # 显示图片
    pyplot.axis('off')  # 不显示坐标轴
    pyplot.show()


def predict(test_X, scaler):
    model = load_model('model_weather')
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]  # 预测数据
    # invert scaling for actual
    inv_y = scaler.inverse_transform(test_X)
    inv_y = inv_y[:, 0]  # 实际数据
    np.set_printoptions(threshold=np.inf)
    # calculate RMSE

    # TODO6:计算均方根误差
    # 说明：计算均方根误差，存入变量rmse
    # 提示：1.计算方法请参照文档5.2.14
    #
    # ====================================

    print('模型和真实值之间的均方根误差RMSE为: %.3f' % rmse)
    print("\n\n模型构建的程序结束，可以运行预测代码")

def clean_clicked(ui):
    data_set = read_raw()
    ui.textBrowser.setText('数据清洗完成')
    ui.textBrowser.setText('数据清洗后前5行为' + str(data_set.head(5)))


def pl_clicked(ui):
    ui.textBrowser.setText('开始绘制图像')
    preview()
    drow_pollution()  # 根据pollution数据集画出各项指标


def train_clicked(ui):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    train_epochs = ui.lineEdit.text()
    train_batch_size = ui.lineEdit_2.text()
    reframed, scaler = cs_to_sl()
    train, test, train_X, train_y, test_X, test_y = train_test(reframed)
    fit_network(train_X, train_y, test_X, test_y, scaler, train_epochs, train_batch_size)
    predict(test_X, scaler)
    pass


def pre_clicked(ui):
    model = tf.keras.models.load_model('model_weather')
    dataset = pd.read_csv('forecast_pollution.csv', header=0, index_col=0)
    print("加载模型model_weather,加载数据集forecast_pollution.csv\n\n")
    # 数据预处理：
    values = dataset.values
    # print(values)
    # LabelEncoder是对不连续的数字或文本编号。
    encoder = LabelEncoder()
    values[:, 4] = encoder.fit_transform(values[:, 4])
    # ensure all data is float
    values = values.astype('float32')
    values_backup = values
    # 数据归一化：此时已经去掉时间值，第一列为污染指数PM2.5:
    scaler = MinMaxScaler(feature_range=(0, 1))
    values = scaler.fit_transform(values)
    train_X = values.reshape((values.shape[0], 1, values.shape[1]))
    print("数据形状已经转换成：", train_X.shape)
    print("方便后期喂入模型进行训练，模型要求的数据格式是（1，8）\n\n")
    # 数据预测：
    yhat = model.predict(train_X)
    print("预测完毕\n\n")
    # 数据还原：
    test_X = train_X.reshape((train_X.shape[0], train_X.shape[2]))
    print("test_X数组已经转换成正常的二维数组，形状为：", test_X.shape)
    # invert scaling for forecast concatenate：数据拼接
    inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    print("数据拼接完毕")
    inv_yhat = scaler.inverse_transform(inv_yhat)
    print("已经转换成正常数值，最终48小时内的污染情况结果如下：")
    inv_yhat = inv_yhat[:, 0]
    print(inv_yhat)
    print("\n\n画出预测情况和实际情况的曲线")
    # 画出预测的污染曲线和实际的污染曲线
    pyplot.figure(num='未来48小时的效果,横轴表示小时，纵轴表示PM2.5的指数', figsize=(10, 10))
    pyplot.plot(values_backup[:, 0], label='actual')
    pyplot.plot(inv_yhat, label='forecast')
    pyplot.legend()
    pyplot.show()
    ui.textBrowser.setText('预测结果为' + str(inv_yhat))


if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_Main_Windows()
    ui.setupUi(MainWindow)
    MainWindow.show()
    ui.pushButton.clicked.connect(partial(train_clicked, ui))
    ui.pushButton_2.clicked.connect(partial(pre_clicked, ui))
    ui.pushButton_3.clicked.connect(partial(clean_clicked, ui))
    ui.pushButton_4.clicked.connect(partial(pl_clicked, ui))
    sys.exit(app.exec_())
