# @File : data_distribution.py 
# @Time : 2019/10/12 
# @Email : jingjingjiang2017@gmail.com
import os
import math
import time
import pandas as pd
import numpy as np

from data.adl import make_dataset
# from data.adl_sequence import make_sequence_dataset
from opt import *

train_video_id = train_video_id + val_video_id


def generate_next_active_object(df, annos):
    index0_1 = []
    index0_1.append(df.index[0])
    index1_0 = []
    index1_0.append(df.index[0])
    for i in range(len(df) - 1):
        if (df.iloc[i, 6] == 0) & (df.iloc[i + 1, 6] == 1):
            index0_1.append(df.index[i + 1])
            # index0_1.append(df.index[i])
        if (df.iloc[i, 6] == 1) & (df.iloc[i + 1, 6] == 0):
            index1_0.append(df.index[i + 1])
    
    if len(index0_1) > 1:
        if df.iloc[0, 6] == 0:
            for i in range(len(index0_1) - 1):
                # print(f'{index0_1[i + 1]}/{index1_0[ i + 1]}')
                annos.loc[(annos.index >= index1_0[i]) & (
                        annos.index <= index0_1[i + 1]), 'is_next_active'] = 1
        if df.iloc[0, 6] == 1:
            for i in range(len(index0_1) - 1):
                annos.loc[(annos.index >= index1_0[i + 1]) & (
                        annos.index <= index0_1[i + 1]), 'is_next_active'] = 1


def generate_bbox(df):
    # bbox = [df.x1, df.y1, df.x2, df.y2, df.is_next_active, df.object_label]
    bbox = [df.x1, df.y1, df.x2, df.y2]
    return bbox


def make_sequence_dataset(args):
    assert args.mode in ['train', 'val', 'test']
    
    print(f'start load {args.mode} data!')
    # start0 = time.process_time()
    items = []
    if args.mode == 'train':
        for video_id in sorted(train_video_id):
            start = time.process_time()
            img_path = os.path.join(args.data_path, frames_path, video_id)
            
            anno_name = 'object_annot_' + video_id + '.txt'
            anno_path = os.path.join(args.data_path, annos_path, anno_name)
            annos = pd.read_csv(anno_path, header=None,
                                delim_whitespace=True, converters={0: str},
                                names=['object_track_id', 'x1', 'y1', 'x2', 'y2',
                                       'frame_id', 'is_active', 'object_label'])
            
            # 判断next_active_object
            annos.insert(loc=7, column='is_next_active', value=0)
            for track_id in sorted(annos.object_track_id.unique()):
                df = annos[annos['object_track_id'] == track_id]
                if df.shape[0] == 1:
                    continue
                if df.shape[0] > 1:
                    if (df[df['is_active'] == 1].shape[0] == 0) | (
                            df[df['is_active'] == 0].shape[0] == 0):
                        continue
                        # print(f'{track_id} are all passive or active!')
                    else:
                        # print(f'df.shape[0]: {df.shape[0]}')
                        generate_next_active_object(df, annos)
            
            annos = annos[annos['is_next_active'] == 1]
            
            for i, idx in enumerate(annos.index):
                df = annos.loc[idx, :]
                # 同一帧图像有多个object, frame_id从0开始, 但ffmpeg得到的图像从1开始
                img_file = img_path + '/' + str(df.frame_id + 1).zfill(6) + '.jpg'
                bbox = generate_bbox(df)  # img bbox
                
                # video_id + '_' + track_id
                item = (img_file, bbox, video_id + '_' + df.object_track_id,
                        df.object_label)
                items.append(item)
            
            end = time.process_time()
            print(f'finished video {video_id}, time is {end - start}')
        
        # 生成sequence data
        df_items = pd.DataFrame(items, columns=['img_file', 'bboxes',
                                                'track_id', 'label'])
        for idx, track_id in enumerate(sorted(df_items.track_id.unique())):
            df_items.loc[df_items.track_id == track_id, 'bs_idx'] = str(idx)
        
        # 去掉seq长度为1的数据
        idx = df_items.bs_idx.value_counts()[
            df_items.bs_idx.value_counts() == 1].index
        df_items = df_items[(~df_items.index.isin(idx))]
        print('================================================================')
        return df_items
    
    if args.mode == 'val':
        for video_id in val_video_id:
            start = time.process_time()
            img_path = os.path.join(args.data_path, frames_path, video_id)
            
            anno_name = 'object_annot_' + video_id + '.txt'
            anno_path = os.path.join(args.data_path, annos_path, anno_name)
            annos = pd.read_csv(anno_path, header=None,
                                delim_whitespace=True, converters={0: str},
                                names=['object_track_id', 'x1', 'y1', 'x2', 'y2',
                                       'frame_id', 'is_active', 'object_label'])
            
            # 判断next_active_object
            annos.insert(loc=7, column='is_next_active', value=0)
            for track_id in sorted(annos.object_track_id.unique()):
                df = annos[annos['object_track_id'] == track_id]
                if df.shape[0] == 1:
                    continue
                if df.shape[0] > 1:
                    if (df[df['is_active'] == 1].shape[0] == 0) | (
                            df[df['is_active'] == 0].shape[0] == 0):
                        continue
                        # print(f'{track_id} are all passive or active!')
                    else:
                        # print(f'df.shape[0]: {df.shape[0]}')
                        generate_next_active_object(df, annos)
            
            annos = annos[annos['is_next_active'] == 1]
            
            for i, idx in enumerate(annos.index):
                df = annos.loc[idx, :]
                # 同一帧图像有多个object, frame_id从0开始, 但ffmpeg得到的图像从1开始
                img_file = img_path + '/' + str(df.frame_id + 1).zfill(6) + '.jpg'
                bbox = generate_bbox(df)  # img bbox
                
                # video_id + '_' + track_id
                item = (img_file, bbox, video_id + '_' + df.object_track_id,
                        df.object_label)
                items.append(item)
            
            end = time.process_time()
            print(f'finished video {video_id}, time is {end - start}')
        
        # 生成sequence data
        df_items = pd.DataFrame(items, columns=['img_file', 'bboxes',
                                                'track_id', 'label'])
        for idx, track_id in enumerate(sorted(df_items.track_id.unique())):
            df_items.loc[df_items.track_id == track_id, 'bs_idx'] = str(idx)
        
        # 去掉seq长度为1的数据
        idx = df_items.bs_idx.value_counts()[
            df_items.bs_idx.value_counts() == 1].index
        df_items = df_items[(~df_items.index.isin(idx))]
        print('================================================================')
        return df_items
    
    if args.mode == 'test':
        for video_id in test_video_id:
            start = time.process_time()
            img_path = os.path.join(args.data_path, frames_path, video_id)
            
            anno_name = 'object_annot_' + video_id + '.txt'
            anno_path = os.path.join(args.data_path, annos_path, anno_name)
            annos = pd.read_csv(anno_path, header=None,
                                delim_whitespace=True, converters={0: str},
                                names=['object_track_id', 'x1', 'y1', 'x2', 'y2',
                                       'frame_id', 'is_active', 'object_label'])
            
            # 判断next_active_object
            annos.insert(loc=7, column='is_next_active', value=0)
            for track_id in sorted(annos.object_track_id.unique()):
                df = annos[annos['object_track_id'] == track_id]
                if df.shape[0] == 1:
                    continue
                if df.shape[0] > 1:
                    if (df[df['is_active'] == 1].shape[0] == 0) | (
                            df[df['is_active'] == 0].shape[0] == 0):
                        continue
                        # print(f'{track_id} are all passive or active!')
                    else:
                        # print(f'df.shape[0]: {df.shape[0]}')
                        generate_next_active_object(df, annos)
            
            annos = annos[annos['is_next_active'] == 1]
            
            for i, idx in enumerate(annos.index):
                df = annos.loc[idx, :]
                # 同一帧图像有多个object, frame_id从0开始, 但ffmpeg得到的图像从1开始
                img_file = img_path + '/' + str(df.frame_id + 1).zfill(6) + '.jpg'
                bbox = generate_bbox(df)  # img bbox
                
                # video_id + '_' + track_id
                item = (img_file, bbox, video_id + '_' + df.object_track_id,
                        df.object_label)
                items.append(item)
            
            end = time.process_time()
            print(f'finished video {video_id}, time is {end - start}')
        
        # 生成sequence data
        df_items = pd.DataFrame(items, columns=['img_file', 'bboxes',
                                                'track_id', 'label'])
        for idx, track_id in enumerate(sorted(df_items.track_id.unique())):
            df_items.loc[df_items.track_id == track_id, 'bs_idx'] = str(idx)
        
        # 去掉seq长度为1的数据
        idx = df_items.bs_idx.value_counts()[
            df_items.bs_idx.value_counts() == 1].index
        df_items = df_items[(~df_items.index.isin(idx))]
        print('================================================================')
        return df_items


def main():
    # train + val
    # data_adl = make_dataset(args)
    # df_adl_ = pd.DataFrame(data_adl, columns=['img_file', 'bbox'])
    df_adl = make_sequence_dataset(args)
    
    print(df_adl.shape)


if __name__ == '__main__':
    main()
