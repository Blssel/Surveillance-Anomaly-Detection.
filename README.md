# Surveillance-Anomaly-Detection.
## 提取特征
### 生成相关索引文件和文件夹
开始之前，需要使用C3D网络提取视频特征。我们使用![C3D官方提供的C3D代码](https://github.com/facebook/C3D/tree/master/C3D-v1.0)进行特征提取。步骤如下：
- 先运行Surveillance-Anomaly-Detection./feature_extract下的gen_io_list.py文件，生成C3D特征提取所必需的索引文件，分别代表输入索引和对应输出存放位置，同时该python文件还负责生成和Videos下文件结构一致的空文件目录，以便存放提取出来的特征。需要先新建UCF_Crimes_C3D_features/Videos文件夹，和./feature_extract/ucfcrime_input_list_video.txt ./feature_extract/ucfcrime_output_list_video_prefix两个文件。请使用如下的方式运行。
```shell
python ./feature_extract/gen_io_list.py video_path frames_path output_path input_file output_file
# 比如
python ./feature_extract/gen_io_list.py /extra_disk/dataset/UCF_Crimes/Videos /extra_disk/dataset/yzy/UCF_Crimes_frames /extra_disk/dataset/yzy/UCF_Crimes_C3D_features/Videos ./feature_extract/ucfcrime_input_list_video.txt ./feature_extract/ucfcrime_output_list_video_prefix.txt
```
- 现将路径切换到C3D_HOME/C3D-v1.0下，依照C3D的说明运行
