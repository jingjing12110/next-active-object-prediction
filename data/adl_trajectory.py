# @File : adl_trajectory.py 
# @Time : 2019/8/10 
# @Email : jingjingjiang2017@gmail.com 

import os
import pandas as pd
import numpy as np
import time
import random
from opt import *


def compute_frame_DTi(df0, df1):
    xys0 = np.zeros([len(df0), 3])
    for idx in range(df0.shape[0]):
        xc = (df0.iloc[idx].x1 + df0.iloc[idx].x2) / 2.0
        yc = (df0.iloc[idx].y1 + df0.iloc[idx].y2) / 2.0
        s = (df0.iloc[idx].x1 - df0.iloc[idx].x2) * (
                    df0.iloc[idx].y1 - df0.iloc[idx].y2)
        xys0[idx] = [xc / 633., yc / 476., s / 306081.]

    xys1 = np.zeros([len(df1), 3])
    for idx in range(df1.shape[0]):
        xc = (df1.iloc[idx].x1 + df1.iloc[idx].x2) / 2.0
        yc = (df1.iloc[idx].y1 + df1.iloc[idx].y2) / 2.0
        s = (df1.iloc[idx].x1 - df1.iloc[idx].x2) * (
                    df1.iloc[idx].y1 - df1.iloc[idx].y2)
        xys1[idx] = [xc / 633., yc / 476., s / 306081.]

    data = np.empty((0, 10))
    bbox = []
    for i, object_id in enumerate(df1.object_track_id):
        xys1_ = xys1[i]

        idx = df0[df0['object_track_id'] == object_id].index.tolist()

        if idx:
            xys0_ = xys0[idx[0]]

            delta_xc = xys1_[0] - xys0_[0]
            delta_yc = xys1_[1] - xys0_[1]
            delta_sc = xys1_[2] - xys0_[2]

            DTi = [xys0_[0], xys0_[1], xys0_[2],
                   xys1_[0], xys1_[1], xys1_[2],
                   delta_xc, delta_yc, delta_sc,
                   df1.iloc[i, 7]]
            data = np.vstack((data, np.array(DTi).reshape((-1, 10))))

            box = [df1.loc[i, 'x1'], df1.loc[i, 'y1'],
                   df1.loc[i, 'x2'], df1.loc[i, 'y2'],
                   df1.loc[i, 'frame_id']]
            bbox.append(box)

    return data, bbox


def compute_DTi(df, is_passive=True):
    # results = []
    xys = np.zeros([len(df), 3])
    for idx in range(df.shape[0]):
        xc = (df.iloc[idx].x1 + df.iloc[idx].x2) / 2.0
        yc = (df.iloc[idx].y1 + df.iloc[idx].y2) / 2.0
        s = (df.iloc[idx].x1 - df.iloc[idx].x2) * (
                    df.iloc[idx].y1 - df.iloc[idx].y2)
        xys[idx] = [xc / 633., yc / 476., s / 306081.]  # ?
        # xys[idx] = [xc, yc, s]

    delta_xys = np.zeros([len(df) - 1, 3])
    for i in range(df.shape[0] - 1):
        delta_xc = xys[i + 1, 0] - xys[i, 0]
        delta_yc = xys[i + 1, 1] - xys[i, 1]
        delta_sc = xys[i + 1, 2] - xys[i, 2]
        delta_xys[i] = [delta_xc, delta_yc, delta_sc]

    # reshape和combine
    DTis = []
    if is_passive:
        # 连续两帧采样
        for index in random.sample(range(len(df) - 1), 1):
            temp = np.vstack(
                [xys[index:index + 2, :], delta_xys[index, :]]).reshape(-1)
            DTis.append(list(temp))
    else:
        for idx in range(len(df) - 1):
            if (df.iloc[idx, 6] == 0) & (df.iloc[idx + 1, 6] == 1):
                temp = np.vstack(
                    [xys[idx:idx + 2, :], delta_xys[idx, :]]).reshape(-1)
                DTis.append(list(temp))

    return np.array(DTis)


def make_dataset(args):
    assert args.mode in ['train', 'val', 'test']

    print(f'start load {args.mode} data!')

    active_trajectory = np.empty((0, 9))
    passive_trajectory = np.empty((0, 9))

    if args.mode == 'train':
        for video_id in train_video_id:
            start = time.process_time()

            anno_name = 'nao_' + video_id + '.txt'
            anno_path = os.path.join(args.data_path, annos_path_v2, anno_name)
            annos = pd.read_csv(anno_path, header=None,
                                delim_whitespace=True, converters={0: str},
                                names=['object_track_id', 'x1', 'y1', 'x2', 'y2',
                                       'frame_id', 'is_active', 'object_label',
                                       'is_next_active'])

            # annos = annos[annos['is_next_active'] == 1]

            annos['object_tracks'] = ['mixed'] * len(annos)

            for i, object_track_id in enumerate(annos.object_track_id.unique()):
                df = annos[annos['object_track_id'] == object_track_id]
                if df.shape[0] > 2:
                    # 全为passive
                    if df[df['is_active'] == 1].shape[0] == 0:
                        annos.loc[annos['object_track_id'] == object_track_id,
                                  'object_tracks'] = 'passive'
                        # 得到passive_trajectory
                        data = compute_DTi(df, is_passive=True)
                        passive_trajectory = np.vstack((passive_trajectory, data))

                    elif df[df['is_active'] == 0].shape[0] == 0:  # 全为active
                        # print(f'discard object_track_id: {object_track_id}')
                        annos.loc[annos['object_track_id'] == object_track_id,
                                  'object_tracks'] = 'discard'

                    else:
                        # active_trajectory
                        data = compute_DTi(df, is_passive=False)
                        # print(data.shape)
                        if len(data) > 0:
                            # print(df.object_track_id.iloc[0])
                            active_trajectory = np.vstack((active_trajectory, data))

                else:
                    continue

            end = time.process_time()
            print(f'finished video {video_id}, time is {end - start}')

        active_label = np.ones(len(active_trajectory))
        active_trajectory = np.column_stack((active_trajectory, active_label))
        passive_label = np.zeros(len(passive_trajectory))
        passive_trajectory = np.column_stack((passive_trajectory, passive_label))
        print('================================================================')

        return active_trajectory, passive_trajectory

    if args.mode == 'val':
        for video_id in val_video_id:
            start = time.process_time()

            anno_name = 'nao_' + video_id + '.txt'
            anno_path = os.path.join(args.data_path, annos_path_v2, anno_name)
            annos = pd.read_csv(anno_path, header=None,
                                delim_whitespace=True, converters={0: str},
                                names=['object_track_id', 'x1', 'y1', 'x2', 'y2',
                                       'frame_id', 'is_active', 'object_label',
                                       'is_next_active'])

            # annos = annos[annos['is_next_active'] == 1]

            annos['object_tracks'] = ['mixed'] * len(annos)

            for i, object_track_id in enumerate(annos.object_track_id.unique()):
                df = annos[annos['object_track_id'] == object_track_id]
                if df.shape[0] > 2:
                    # 全为passive
                    if df[df['is_active'] == 1].shape[0] == 0:
                        annos.loc[annos['object_track_id'] == object_track_id,
                                  'object_tracks'] = 'passive'
                        # 得到passive_trajectory
                        data = compute_DTi(df, is_passive=True)
                        passive_trajectory = np.vstack((passive_trajectory, data))

                    elif df[df['is_active'] == 0].shape[0] == 0:  # 全为active
                        # print(f'discard object_track_id: {object_track_id}')
                        annos.loc[annos['object_track_id'] == object_track_id,
                                  'object_tracks'] = 'discard'

                    else:
                        # active_trajectory
                        data = compute_DTi(df, is_passive=False)
                        # print(data.shape)
                        if len(data) > 0:
                            # print(df.object_track_id.iloc[0])
                            active_trajectory = np.vstack((active_trajectory, data))

                else:
                    continue

            end = time.process_time()
            print(f'finished video {video_id}, time is {end - start}')

        active_label = np.ones(len(active_trajectory))
        active_trajectory = np.column_stack((active_trajectory, active_label))
        passive_label = np.zeros(len(passive_trajectory))
        passive_trajectory = np.column_stack((passive_trajectory, passive_label))
        print('================================================================')

        return active_trajectory, passive_trajectory

    if args.mode == 'test':
        test_data = np.empty((0, 10))
        bboxs = np.empty((0, 5))

        for video_id in test_video_id:
            start = time.process_time()
            anno_name = 'nao_' + video_id + '.txt'
            anno_path = os.path.join(args.data_path, annos_path_v2, anno_name)
            annos = pd.read_csv(anno_path, header=None,
                                delim_whitespace=True, converters={0: str},
                                names=['object_track_id', 'x1', 'y1', 'x2', 'y2',
                                       'frame_id', 'is_active', 'object_label',
                                       'is_next_active'])

            # annos = annos[annos['is_next_active'] == 1]

            frame_ids = np.sort(annos.frame_id.unique())
            for idx in range(len(frame_ids) - 1):
                df0 = annos[annos['frame_id'] == frame_ids[idx]].reset_index()
                df1 = annos[annos['frame_id'] == frame_ids[idx + 1]].reset_index()

                data, bbox = compute_frame_DTi(df0, df1)
                if len(bbox) > 0:
                    # print(f'data: {len(data)}; bbox: {len(bbox)}')
                    test_data = np.vstack((test_data, data))
                    bboxs = np.vstack((bboxs, np.array(bbox)))

                # 得到df1对应bbox的大小

            end = time.process_time()
            print(f'finished video {video_id}, time is {end - start}')
        print('================================================================')

        return test_data, bboxs


if __name__ == '__main__':

    args.mode = 'test'
    data = make_dataset(args=args)

