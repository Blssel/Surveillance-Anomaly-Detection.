#coding:UTF-8

__author__='Zhiyu Yin'

import tensorflow as tf
import binfeature_read as bread

from arguments import Args

'''
该模块负责返回Dataset类对象
'''
def _read_split(split_path):
  # 读入训练列表
  with open(split_path,'r') as fout:
    train_list=fout.read().split('\n')
  ano_list=[]
  no_list=[]
  for item in train_list:
    if item.split('/')[0]=='Training_Normal_Videos_Anomaly':
      no_list.append(item)
    else:
      ano_list.append(item)
  ano_list.remove('')

  return ano_list, no_list
def _parse_function(filename):
  # 定义容器，存放该视频的所有feature，
  feature_bag=[]
  # 获取特征名称list
  feature_path=os.path.join(Args.feature_path , filename.split('.')[0])
  feature_list=glob.glob(os.path.join(feature_path_ano,'*'))
  ##读取该视频对应的特征文件（很多个）
  ##seg=np.array([])
  for index in range(0,len(feature_list)-1,2):
    header1, data1 =  bread.read_binary_fc(os.path.join(feature_path, feature_list[index].split('/')[-1]))
    header2, data2 =  bread.read_binary_fc(os.path.join(feature_path, feature_list[index+1].split('/')[-1]))
    feature_bag.append((data1+data2)/2.0)
  if len(feature_list)%2==1: #如果特征是奇数个，则最后一个（标号为len(feature_list)-1,或者-1）需要单独处理
    header, data =  bread.read_binary_fc(os.path.join(feature_path, feature_list[-1].split('/')[-1]))
    feature_bag.append(data)
  # 将此feature_bag过fc层并筛选出最大值返回给dataset
  feature_bag=np.array(feature_bag)
  return np.array(feature_bag)
  

def get_dataset():
  # 从txt文件中读入split列表
  split_list_ano,split_list_no=_read_split(Args.train_split_path)
  # 创建两个dataset
  dataset_ano=tf.data.Dataset.from_tensor_slice(split_list_ano)
  dataset_no=tf.data.Dataset.from_tensor_slice(split_list_no)
  # 映射
  dataset_ano=dataset_ano.map(_parse_function)
  dataset_no=dataset_no.map(_parse_function)
  dataset_ano=dataset_ano.repeat().batch(Args.batch_size)
  dataset_no=dataset_no.repeat().batch(Args.batch_size)
  return dataset_ano,dataset_no

def _to_one_dim(high_dim_list):
  '''
  将列表降维成一维列表
  '''
  list_str=str(high_dim_list)
  list_str=list_str.replace('[','')
  list_str=list_str.replace(']','')
  # 再变回list
  one_dim_list=list(eval(list_str))
  return one_dim_list

def next_batch():
  dataset_ano,dataset_no=get_dataset()
  # 创建迭代器
  iter_ano = dataset_ano.make_one_shot_iterator()
  iter_no = dataset_no.make_one_shot_iterator()
  # 读取batch
  ano_batch=iter_ano.get_next()
  no_batch=iter_no.get_next()
  # 重点解释：由于batch中每一项是一个feature_bag，为了卷积运算的方便，需要去掉feature_bag的外层[]，同时为了后面使用tf.segment_max函数，需要生成segment_ids numpy数组
  ano_segment_ids=[]
  no_segment_ids=[]
  for i in range(Args.batch_size):
    ano_segment_ids.append([i]* ano_batch[i].get_shape().as_list()[0])
    no_segment_ids.append([i]* no_batch[i].get_shape().as_list()[0])
  ano_segment_ids=np.array(_to_one_dim(ano_segment_ids))
  no_segment_ids=np.array(_to_one_dim(no_segment_ids))
  # 将batch中多余维度去掉（第一维）
  # 返回
  return ano_batch, ano_segment_ids, no_batch, no_segment_ids




