# @File : adl_v0(未做sequence删除处理).py
# @Time : 2019/7/26 
# @Email : jingjingjiang2017@gmail.com
import os
import shutil
import pandas as pd

data_path = '/home/kaka/SSD_data/ADL'
annos_path = 'ADL_annotations/object_annotation'
action_path = 'ADL_annotations/action_annotation'
frames_path = 'ADL_frames'
key_frames_path = 'ADL_key_frames'

video_id = {'P_01', 'P_02', 'P_03', 'P_04', 'P_05', 'P_06', 'P_07',
            'P_08', 'P_09', 'P_10', 'P_11', 'P_12', 'P_13', 'P_14',
            'P_15', 'P_16', 'P_17', 'P_18', 'P_19', 'P_20'}
# video_id = {'P_19'}

for vi in video_id:
    img_path = os.path.join(data_path, frames_path, vi)
    key_img_path = os.path.join(data_path, key_frames_path, vi)
    
    anno_name = 'object_annot_' + vi + '.txt'
    anno_path = os.path.join(data_path, annos_path, anno_name)
    annos = pd.read_csv(anno_path, header=None,
                        delim_whitespace=True, converters={0: str},
                        names=['object_track_id', 'x1', 'y1', 'x2', 'y2',
                               'frame_id', 'is_active', 'object_label'])
    
    print(f"{vi}: annos_{annos.shape[0]}, "
          f"annos_unique_{annos.frame_id.unique().shape[0]}")
    
    for frame_id in annos.frame_id.unique():
        img_file = img_path + '/' + str(frame_id + 1).zfill(6) + '.jpg'
        
        shutil.copy(img_file, os.path.join(key_img_path,
                                           str(frame_id + 1).zfill(6) + '.jpg'))
        
    print(f"finished {vi}!")
