# @File : data_anno.py 
# @Time : 2019/7/27 
# @Email : jingjingjiang2017@gmail.com 
import numpy as np
import pandas as pd
import os

data_root = '/media/kaka/Jing/ADL'
annos_path = 'ADL_annotations/object_annotation'
frames_path = 'ADL_frames'

if __name__ == '__main__':
    img_path = os.path.join(data_root, frames_path)

    for sub_file in os.listdir(img_path):
        frame_path = os.path.join(img_path, sub_file)

        anno_name = 'object_annot_' + sub_file + '.txt'
        anno_path = os.path.join(data_root, annos_path, anno_name)
        annos = pd.read_csv(anno_path, header=None, delim_whitespace=True,
                            converters={0: str},
                            names=['object_track_id', 'x1', 'y1', 'x2', 'y2',
                                   'frame_id', 'is_active', 'object_label'])
        # annos['video_id'] = sub_file
        # with open(anno_path, 'r') as f_read:
        #     annos = f_read.readlines()
        # annos = [a.split(' ') for a in annos]
        annos_active = annos[annos['is_active'] == 1]
        # annos_active.frame_number.count() > annos_active.frame_number.nunique()
        # annos_active.['frame_number'].value_counts()
        # 说明同一帧图像可能有几个active object
        annos_passive = annos[annos['is_active'] == 0]
        if annos_passive.shape[0] > annos_active.shape[0]:
            passive_number = annos_active.shape[0]
            annos_passive = annos_passive.sample(n=passive_number)

        anno_frames_name = 'object_annot_' + sub_file + '_annotated_frames.txt'
        anno_frames_path = os.path.join(data_root, annos_path, anno_frames_name)
        anno_frames = pd.read_csv(anno_frames_path, header=None,
                                  names=['frame_id'])




