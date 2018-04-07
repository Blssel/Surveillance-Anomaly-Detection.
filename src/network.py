#coding:UTF-8

__author__='Zhiyu Yin'

import tensorflow as tf
'''
全连接层网络参数定义及前向传播
'''

class fc_layer:
  def __inti__(self,dropout,batch_size):
    # 属性，包括dropout，batch_size
    self.dropout=dropout
    self.batch_size=batch_size
    # 参数变量
    with tf.variable_scope('fc') as var_scope:
      weights = {
                'w1': _variable_with_weight_decay('w1', [4096, 512], 0.0005),
                'w2': _variable_with_weight_decay('w2', [512,32], 0.0005),
                'w3': _variable_with_weight_decay('w3', [32,1], 0.0005),
                }
      biases = {
                'b1': _variable_with_weight_decay('b1', [512], 0.000),
                'b2': _variable_with_weight_decay('b2', [32], 0.000),
                }

  def _variable_on_cpu(self,name, shape, initializer):
    with tf.device('/cpu:0'):
      var = tf.get_variable(name, shape, initializer=initializer)
    return var
  
  
  def _variable_with_weight_decay(self,name, shape, wd):
    var = _variable_on_cpu(name, shape, tf.contrib.layers.xavier_initializer())
    if wd is not None:
      weight_decay = tf.nn.l2_loss(var)*wd
      tf.add_to_collection('weightdecay_losses', weight_decay)
    return var

  def fc_inference(self,_input):
    fc_7 = tf.matmul(_input, self.weights['w1']) + self.biases['b1']
    fc_7 = tf.nn.relu(fc_7, name='fc7') # Relu activation
    fc_7 = tf.nn.dropout(fc_7, self.dropout)  

    fc_8 = tf.matmul(fc_7, self.weights['w2']) + self.biases['b2']
    fc_8 = tf.nn.sigmoid(fc_8, name='fc8') # Relu activation
    fc_8 = tf.nn.dropout(fc_8, self.dropout)  

    out=tf.matmul(fc_8, self.weights['w3'])

    return out
