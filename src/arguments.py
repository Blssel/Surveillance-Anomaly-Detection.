#coding:UTF-8

__author__='Zhiyu Yin'

'''
存放参数
'''
class Args
  batch_size=30
  max_steps=8000
  dropout=
  lamb1=8e-5
  lamb2=8e-5

  train_split_path='/extra_disk/dataset/UCF_Crimes/Anomaly_Detection_splits/Anomaly_Train.txt'
  #train_split_path='/share/dataset/UCF_Crimes/Anomaly_Detection_splits/Anomaly_Train.txt'
  test_split_path='/extra_disk/dataset/UCF_Crimes/Anomaly_Detection_splits/Anomaly_Test.txt'
  feature_path='/home/zy_17/workspace/C3D/C3D-v1.0/UCF_Crimes_C3D_features/Videos'

  model_save_dir='models/ucf-crimes'

  # 比较特殊： 由于dataset映射的过程中必须保持每个视频的segment的数量一致（否则无法组成batch），所以需要padding
  max_segment=1000
  def __inti__():
    
