#coding:utf-8
import os
import cv2
import argparse
    
def parse_args():
  parser=argparse.ArgumentParser(description='Process path')
  parser.add_argument('video_path',help='video path,for example, /share/dataset/UCF_Crimes/Videos')
  parser.add_argument('frames_path',help='corresponding frame path,for example, /share/dataset/UCF_Crimes/frames')
  parser.add_argument('output_path',help='path saving features,for example, /share/dataset/UCF_Crimes_C3D_features/Videos')
  parser.add_argument('input_file',help='a txt file each item of whom represent the beginning frame of a clip')
  parser.add_argument('output_file',help='a txt file each item of whom represent the name of the feature corresponding to the item in input_file')
  return parser.parse_args()
  
def main():
  #解析参数
  args=parse_args()
  video_path=args.video_path
  frames_path=args.frames_path
  output_path=args.output_path
  input_file=args.input_file
  output_file=args.output_file
  #打开文件  
  f1=open(input_file,'w')
  f2=open(output_file,'w')

  # 遍历每个文件
  for cur_location,dir_names,file_names in os.walk(video_path):
    if file_names==None:
      continue
    else:
      # 对每一个视频
      action_name=cur_location.split('/')[-1]
      for vid in file_names:
        # 先为该视频创建对应输出文件夹
        outdir=os.path.join(output_path,action_name,vid.split('.')[0])
        if not os.path.exists(outdir):
          os.makedirs(outdir)
        # 读取视频，并获取总帧数
        cap=cv2.VideoCapture(os.path.join(cur_location,vid))
        num_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 输入文件三元组
        vid_fullpath=os.path.join(frames_path,action_name,vid.split('.')[0])
        for frame_index in range(num_frames):
          if frame_index%16==0 and (frame_index+15)<num_frames:
            start_frame=frame_index
            f1.write(vid_fullpath+' '+str(start_frame+1)+' '+'0'+'\n')
            # 输出文件一元组
            f2.write(os.path.join(outdir,'%06d'%frame_index)+'\n')
  f1.close()  
  f2.close()  
if __name__=='__main__':
  main()
