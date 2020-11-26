# @File : adl.py
# @Time : 2019/9/23 
# @Email : jingjingjiang2017@gmail.com 

import math
import os
import pickle
import time
from ast import literal_eval

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F_trans
from PIL import Image
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from opt import *


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
    bbox = [df.x1, df.y1, df.x2, df.y2, df.is_next_active]
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

        # 生成带bs_idx的数据
        df_items = pd.DataFrame(items, columns=['img_file', 'bboxes',
                                                'track_id', 'label'])
        del items

        for idx, track_id in enumerate(sorted(df_items.track_id.unique())):
            df_items.loc[df_items.track_id == track_id, 'bs_idx'] = str(idx)

        # 去掉seq长度为1的数据(bs_idx的个数=1的bs_idx是)
        bs_idxs_ = df_items.bs_idx.value_counts()[
            df_items.bs_idx.value_counts() == 1].index
        df_items = df_items[(~df_items.bs_idx.isin(bs_idxs_))].reset_index(
            drop=True)
        # seq长度大于10的数据，cut off掉-10以前的数据
        bs_idxs_ = df_items.bs_idx.value_counts()[
            df_items.bs_idx.value_counts() >= 10].index  # 长度大于10的bs_idx
        df_items_tmp = df_items[(df_items.bs_idx.isin(bs_idxs_))].reset_index(
            drop=True)  # 长度大于10的数据
        df_items = df_items[(~df_items.bs_idx.isin(bs_idxs_))].reset_index(
            drop=True)  # 长度小于10的数据
        for bs_idx in bs_idxs_:
            df_tmp = df_items_tmp[df_items_tmp.bs_idx == bs_idx].reset_index(
                drop=True)
            df_items = pd.concat([df_items, df_tmp[-8:]], ignore_index=True)

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

        # 生成
        df_items = pd.DataFrame(items, columns=['img_file', 'bboxes',
                                                'track_id', 'label'])
        del items

        for idx, track_id in enumerate(sorted(df_items.track_id.unique())):
            df_items.loc[df_items.track_id == track_id, 'bs_idx'] = str(idx)

        # 去掉seq长度为1的数据(bs_idx的个数=1的bs_idx是)
        bs_idxs_ = df_items.bs_idx.value_counts()[
            df_items.bs_idx.value_counts() == 1].index
        df_items = df_items[(~df_items.bs_idx.isin(bs_idxs_))].reset_index(
            drop=True)
        # seq长度大于10的数据，cut off掉-10以前的数据
        bs_idxs_ = df_items.bs_idx.value_counts()[
            df_items.bs_idx.value_counts() >= 10].index
        df_items_tmp = df_items[(df_items.bs_idx.isin(bs_idxs_))].reset_index(
            drop=True)
        df_items = df_items[(~df_items.bs_idx.isin(bs_idxs_))].reset_index(
            drop=True)
        for bs_idx in bs_idxs_:
            df_tmp = df_items_tmp[df_items_tmp.bs_idx == bs_idx].reset_index(
                drop=True)
            df_items = pd.concat([df_items, df_tmp[-8:]], ignore_index=True)

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
        del items

        for idx, track_id in enumerate(sorted(df_items.track_id.unique())):
            df_items.loc[df_items.track_id == track_id, 'bs_idx'] = str(idx)

        # 去掉seq长度为1的数据(bs_idx的个数=1的bs_idx是)
        bs_idxs_ = df_items.bs_idx.value_counts()[
            df_items.bs_idx.value_counts() == 1].index
        df_items = df_items[(~df_items.bs_idx.isin(bs_idxs_))].reset_index(
            drop=True)
        # seq长度大于10的数据，cut off掉-10以前的数据
        bs_idxs_ = df_items.bs_idx.value_counts()[
            df_items.bs_idx.value_counts() >= 10].index
        df_items_tmp = df_items[(df_items.bs_idx.isin(bs_idxs_))].reset_index(
            drop=True)
        df_items = df_items[(~df_items.bs_idx.isin(bs_idxs_))].reset_index(
            drop=True)
        for bs_idx in bs_idxs_:
            df_tmp = df_items_tmp[df_items_tmp.bs_idx == bs_idx].reset_index(
                drop=True)
            df_items = pd.concat([df_items, df_tmp[-8:]], ignore_index=True)

        print('================================================================')
        return df_items


def make_sequence_dataset_v2(args):
    """自己标注的adl next active object 数据生成器
    """
    assert args.mode in ['train', 'val', 'test']

    print(f'start load {args.mode} data!')
    items = []
    if args.mode == 'train':
        for video_id in sorted(train_video_id):
            start = time.process_time()
            img_path = os.path.join(args.data_path, frames_path, video_id)

            anno_name = 'nao_' + video_id + '.txt'
            anno_path = os.path.join(args.data_path, annos_path_v2, anno_name)
            annos = pd.read_csv(anno_path, header=None,
                                delim_whitespace=True, converters={0: str},
                                names=['object_track_id', 'x1', 'y1', 'x2', 'y2',
                                       'frame_id', 'is_active', 'object_label',
                                       'is_next_active'])

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

        # 生成带bs_idx的数据
        df_items = pd.DataFrame(items, columns=['img_file', 'bboxes',
                                                'track_id', 'label'])
        del items

        for idx, track_id in enumerate(sorted(df_items.track_id.unique())):
            df_items.loc[df_items.track_id == track_id, 'bs_idx'] = str(idx)

        print('================================================================')
        return df_items

    if args.mode == 'val':
        for video_id in val_video_id:
            start = time.process_time()
            img_path = os.path.join(args.data_path, frames_path, video_id)

            anno_name = 'nao_' + video_id + '.txt'
            anno_path = os.path.join(args.data_path, annos_path_v2, anno_name)
            annos = pd.read_csv(anno_path, header=None,
                                delim_whitespace=True, converters={0: str},
                                names=['object_track_id', 'x1', 'y1', 'x2', 'y2',
                                       'frame_id', 'is_active', 'object_label',
                                       'is_next_active'])

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

        # 生成
        df_items = pd.DataFrame(items, columns=['img_file', 'bboxes',
                                                'track_id', 'label'])
        del items

        for idx, track_id in enumerate(sorted(df_items.track_id.unique())):
            df_items.loc[df_items.track_id == track_id, 'bs_idx'] = str(idx)

        print('================================================================')
        return df_items

    if args.mode == 'test':
        for video_id in test_video_id:
            start = time.process_time()
            img_path = os.path.join(args.data_path, frames_path, video_id)

            anno_name = 'nao_' + video_id + '.txt'
            anno_path = os.path.join(args.data_path, annos_path_v2, anno_name)
            annos = pd.read_csv(anno_path, header=None,
                                delim_whitespace=True, converters={0: str},
                                names=['object_track_id', 'x1', 'y1', 'x2', 'y2',
                                       'frame_id', 'is_active', 'object_label',
                                       'is_next_active'])

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
        del items

        for idx, track_id in enumerate(sorted(df_items.track_id.unique())):
            df_items.loc[df_items.track_id == track_id, 'bs_idx'] = str(idx)

        print('================================================================')
        return df_items


class AdlDataset(Dataset):
    def __init__(self, args):
        self.args = args

        # self.data = make_sequence_dataset(args)
        self.data = make_sequence_dataset_v2(args)
        self.data = self.data_process()

        # pandas的shuffle
        self.data = self.data.sample(frac=1).reset_index(drop=True)

        self.transform_img = transforms.Compose([
            # transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        # ImageNet
        self.transform_label = transforms.ToTensor()

    def __getitem__(self, item):
        df_item = self.data.iloc[item, :]

        img = Image.open(df_item.img_file).convert('RGB')
        img = img.resize((640, 480), Image.ANTIALIAS)
        # img.show()

        # bboxes = df_item.bboxes
        mask = self.generate_mask(df_item.bboxes)
        mask = Image.fromarray(255 * mask).convert('L')

        # if self.args.mode == 'train':
        #     img, mask = self.random_scale_crop(img, mask, crop_size=[432, 576])

        img = img.resize((self.args.img_resize[1],
                          self.args.img_resize[0]))
        mask = mask.resize((self.args.img_resize[1],
                            self.args.img_resize[0]))

        # img = self.transform(img)
        # mask = self.transform(mask)
        # img.show()

        img = self.transform_img(img)
        mask = self.transform_label(mask)[0, :, :]

        return img, mask

    def __len__(self):  # batch迭代的次数与其有关
        return self.data.shape[0]

    def generate_mask(self, bbox):
        # mask = np.zeros(self.args.img_size)
        # adl数据集给的标签坐标就是在（480， 640）的分辨率下的
        mask = np.zeros((480, 640), dtype=np.float32)

        # for b in bboxes:
        if bbox[4] == 1:
            mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1

        return mask

    def random_scale_crop(self, img, mask, crop_size, angle=30):
        height, width = crop_size
        # img, rect = transforms.RandomCrop((h, w))(img)
        if random.random() > 0.8:
            i, j, h, w = transforms.RandomCrop.get_params(img, (height, width))
            img_ = F_trans.crop(img, i, j, h, w)
            mask_ = F_trans.crop(mask, i, j, h, w)
        else:
            img_ = img
            mask_ = mask

        # if random.random() > 0.4:
        #     angle_ = transforms.RandomRotation.get_params([-angle, angle])
        #     img_ = F_trans.rotate(img_, angle_, resample=Image.NEAREST)
        #     mask_ = F_trans.rotate(mask_, angle_, resample=Image.NEAREST)
        # else:
        #     img_ = img
        #     mask_ = mask

        return img_, mask_

    def data_process(self):
        df_items = pd.DataFrame(columns=['img_file', 'bboxes', 'track_id',
                                         'label', 'bs_idx'])
        for itm in self.data.bs_idx.unique():
            df_item = self.data[self.data['bs_idx'] == str(itm)]
            if df_item.shape[0] <= 4:
                df_item = df_item
            elif (df_item.shape[0] > 4) & (df_item.shape[0] < 10):
                df_item = df_item.iloc[[0, math.ceil(df_item.shape[0] / 3),
                                        df_item.shape[0] - 1]]
            else:
                df_item = df_item[-8:]
                df_item = df_item.iloc[[0, math.ceil(df_item.shape[0] / 3),
                                        df_item.shape[0] - 1]]
            df_item = df_item.reset_index(drop=True)

            df_items = pd.concat([df_items, df_item], ignore_index=True)

        return df_items


class AdlDatasetLabeled(Dataset):
    def __init__(self, args):
        self.args = args

        self.data = make_sequence_dataset_v2(args)

        # pandas的shuffle
        self.data = self.data.sample(frac=1).reset_index(drop=True)

        self.transform_img = transforms.Compose([
            # transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])  # ImageNet
        self.transform_label = transforms.ToTensor()

    def __getitem__(self, item):
        df_item = self.data.iloc[item, :]

        img = Image.open(df_item.img_file).convert('RGB')
        img = img.resize((640, 480), Image.ANTIALIAS)
        # img.show()

        # bboxes = df_item.bboxes
        mask = self.generate_mask(df_item.bboxes)
        mask = Image.fromarray(255 * mask).convert('L')

        # if self.args.mode == 'train':
        #     img, mask = self.random_scale_crop(img, mask, crop_size=[432, 576])

        img = img.resize((self.args.img_resize[1],
                          self.args.img_resize[0]))
        mask = mask.resize((self.args.img_resize[1],
                            self.args.img_resize[0]))

        # img.show()

        img = self.transform_img(img)
        mask = self.transform_label(mask)[0, :, :]

        return img, mask

    def __len__(self):  # batch迭代的次数与其有关
        return self.data.shape[0]

    def generate_mask(self, bbox):
        # mask = np.zeros(self.args.img_size)
        # adl数据集给的标签坐标就是在（480， 640）的分辨率下的
        mask = np.zeros((480, 640), dtype=np.float32)

        # for b in bboxes:
        if bbox[4] == 1:
            mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1

        return mask

    def random_scale_crop(self, img, mask, crop_size, angle=30):
        height, width = crop_size
        # img, rect = transforms.RandomCrop((h, w))(img)
        if random.random() > 0.8:
            i, j, h, w = transforms.RandomCrop.get_params(img, (height, width))
            img_ = F_trans.crop(img, i, j, h, w)
            mask_ = F_trans.crop(mask, i, j, h, w)
        else:
            img_ = img
            mask_ = mask

        if random.random() > 0.4:
            angle_ = transforms.RandomRotation.get_params([-angle, angle])
            img_ = F_trans.rotate(img_, angle_, resample=Image.NEAREST)
            mask_ = F_trans.rotate(mask_, angle_, resample=Image.NEAREST)
        else:
            img_ = img
            mask_ = mask

        return img_, mask_


class AdlSequenceDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.crop = transforms.RandomCrop((args.img_resize[0],
                                           args.img_resize[1]))
        self.transform_label = transforms.ToTensor()
        self.data = make_sequence_dataset(args)

        if args.normalize:
            self.transform = transforms.Compose([  # [h, w]
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.256, 0.341, 0.393],
                #                      std=[0.212, 0.224, 0.229])
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])  # ImageNet
            ])
        else:
            self.transform = transforms.Compose([  # [h, w]
                # transforms.Resize([args.img_resize[0], args.img_resize[1]]),
                transforms.ToTensor()
            ])

    def __getitem__(self, item):
        df_item = self.data[self.data['bs_idx'] == str(item)]
        # images = torch.FloatTensor(df_item.shape[0], 3, self.args.img_resize[0],
        #                            self.args.img_resize[1])
        # masks = torch.FloatTensor(df_item.shape[0], self.args.img_resize[0],
        #                           self.args.img_resize[1])

        # [sequence_len, channel, H, W]
        images = torch.FloatTensor(3, 3, self.args.img_resize[0],
                                   self.args.img_resize[1])
        masks = torch.FloatTensor(3, self.args.img_resize[0],
                                  self.args.img_resize[1])

        if df_item.shape[0] == 2:
            df_item = df_item.append(df_item[-1:])
        elif (df_item.shape[0] >= 3) & (df_item.shape[0] < 10):
            df_item = df_item.iloc[[0, math.ceil(df_item.shape[0] / 3),
                                    df_item.shape[0] - 1]]
        else:  # df_item.shape[0] >= 10:
            df_item = df_item[-8:]
            df_item = df_item.iloc[[0, math.ceil(df_item.shape[0] / 3),
                                    df_item.shape[0] - 1]]
        df_item = df_item.reset_index(drop=True)

        for i, idx in enumerate(df_item.index):
            img_file = df_item.loc[idx, 'img_file']
            img = Image.open(img_file).convert('RGB')
            img = img.resize((640, 480), Image.ANTIALIAS)

            bboxes = df_item.loc[idx, 'bboxes']

            mask = self.generate_mask(bboxes)
            mask = Image.fromarray(mask)

            img = img.resize((self.args.img_resize[1],
                              self.args.img_resize[0]))
            mask = mask.resize((self.args.img_resize[1],
                                self.args.img_resize[0]))
            img = self.transform(img)
            mask = self.transform_label(mask)
            mask = mask[0, :, :]

            images[i, :, :, :] = img
            masks[i, :, :] = mask

        return images, masks

    def __len__(self):  # batch迭代的次数与其有关
        return self.data.bs_idx.unique().shape[0]

    def generate_mask(self, bbox):
        # mask = np.zeros(self.args.img_size)
        # adl数据集给的标签坐标就是在（480， 640）的分辨率下的
        mask = np.zeros((480, 640), dtype=np.float32)

        # for b in bboxes:
        if bbox[4] == 1:
            mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1

        return mask


class AdlSeq3Dataset(Dataset):
    def __init__(self, args):
        super(AdlSeq3Dataset, self).__init__()
        self.args = args
        self.crop = transforms.RandomCrop((args.img_resize[0],
                                           args.img_resize[1]))
        self.transform_label = transforms.ToTensor()
        # self.data = make_sequence_dataset(args)
        self.data_path = os.path.join(args.data_path, features_path)

        # load data
        self.data = pd.read_csv(os.path.join(
            self.data_path, f'{self.args.mode}_seq_df.csv'))
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        self.features = pickle.load(open(os.path.join(
            self.data_path, f'{self.args.mode}_vgg16_features.pickle'), 'rb'))

        if args.normalize:
            self.transform = transforms.Compose([  # [h, w]
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.256, 0.341, 0.393],
                #                      std=[0.212, 0.224, 0.229])
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])  # ImageNet
            ])
        else:
            self.transform = transforms.Compose([  # [h, w]
                # transforms.Resize([args.img_resize[0], args.img_resize[1]]),
                transforms.ToTensor()
            ])

    def __getitem__(self, item):
        df_item = self.data[self.data['bs_idx'] == item]
        feature12 = self.features[f'{item}']

        # [sequence_len, channel, H, W]
        images = torch.FloatTensor(3, 3, self.args.img_resize[0],
                                   self.args.img_resize[1])
        masks = torch.FloatTensor(3, self.args.img_resize[0],
                                  self.args.img_resize[1])
        # [num, len, channel, feature_size]
        features = torch.FloatTensor(2, self.args.feature_len, 512, 70)
        features[0, :, :, :] = torch.from_numpy(feature12[0])
        features[1, :, :, :] = torch.from_numpy(feature12[1])

        for i, idx in enumerate(df_item.index):
            img_file = df_item.loc[idx, 'img_file']
            img = Image.open(img_file).convert('RGB')
            # img = img.resize((640, 480), Image.ANTIALIAS)

            bboxes = df_item.loc[idx, 'bboxes']
            mask = self.generate_mask(bboxes)
            mask = Image.fromarray(mask)

            img = img.resize((self.args.img_resize[1],
                              self.args.img_resize[0]))
            mask = mask.resize((self.args.img_resize[1],
                                self.args.img_resize[0]))

            img = self.transform(img)
            mask = self.transform_label(mask)[0, :, :]

            images[i, :, :, :] = img
            masks[i, :, :] = mask

        return images, masks, features

    def __len__(self):  # batch迭代的次数与其有关
        return self.data.bs_idx.unique().shape[0]

    @staticmethod
    def generate_mask(bbox):
        mask = np.zeros((480, 640), dtype=np.float32)
        if bbox[4] == 1:
            mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1

        return mask


class AdlDatasetV2(Dataset):
    def __init__(self, args):
        super(AdlDatasetV2, self).__init__()
        self.args = args
        self.crop = transforms.RandomCrop((args.img_resize[0],
                                           args.img_resize[1]))
        self.transform_label = transforms.ToTensor()

        self.data_path = os.path.join(args.data_path, features_path)

        # self.static_hm = self.gen_static_hand_dm()
        self.hand_hms = pickle.load(open(os.path.join(
            self.data_path, f'{self.args.mode}_hand_hms.pkl'), 'rb'))

        # load data
        # self.data = pd.read_csv(os.path.join(
        #     self.data_path, f'adl_{self.args.mode}_df.csv'))
        self.data = pd.read_pickle(os.path.join(
            self.data_path, f'adl_{args.mode}_hand_bbox_df.pkl'))
        self.data['bboxes'] = self.data['bboxes'].apply(
            lambda x: literal_eval(x))

        print(f'{args.mode} data: {self.data.shape[0]}')

        self.data = self.data.sample(frac=1).reset_index(drop=True)

        if args.normalize:
            self.transform = transforms.Compose([  # [h, w]
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.256, 0.341, 0.393],
                #                      std=[0.212, 0.224, 0.229])
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])  # ImageNet
            ])
        else:
            self.transform = transforms.Compose([  # [h, w]
                # transforms.Resize([args.img_resize[0], args.img_resize[1]]),
                transforms.ToTensor()
            ])

    def __getitem__(self, item):
        df_item = self.data.iloc[item, :]

        img_file = df_item.img_file
        img = Image.open(img_file).convert('RGB')
        img = img.resize((640, 480), Image.ANTIALIAS)

        # bboxes = df_item.bboxes
        mask = self.generate_mask(df_item.bboxes)
        mask = Image.fromarray(mask)

        hand_hm = self.hand_hms[img_file]

        img = img.resize((self.args.img_resize[1],
                          self.args.img_resize[0]))
        mask = mask.resize((self.args.img_resize[1],
                            self.args.img_resize[0]))

        img = self.transform(img)
        mask = self.transform_label(mask)[0, :, :]
        hand_hm = self.transform_label(hand_hm)

        return img, mask, hand_hm

    def __len__(self):  # batch迭代的次数与其有关
        return self.data.shape[0]

    def generate_mask(self, bbox):
        # mask = np.zeros(self.args.img_size)
        # adl数据集给的标签坐标就是在（480， 640）的分辨率下的
        mask = np.zeros((480, 640), dtype=np.float32)

        if bbox[4] == 1:
            mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 1

        return mask

    @staticmethod
    def generate_hand_hm(img, hand_bbox):
        im = np.zeros((img.size[0], img.size[1]))

        if len(hand_bbox) > 0:
            points = []
            for box in hand_bbox:
                points.append((box[0], box[1]))
                # points.append((box[0], box[3]))
                points.append((box[1], box[1]))
                # points.append((box[1], box[3]))
            points = np.array(points).transpose()
            im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
            im = ndimage.gaussian_filter(im, sigma=img.size[0] / (
                    4. * points.shape[1]))
            im = (im - im.min()) / (im.max() - im.min())

        return im.transpose()

    @staticmethod
    def gen_static_hand_dm():  # 用平均值
        im = np.zeros((1280, 960))
        points = [(459, 465), (896, 465)]

        points = np.array(points).transpose()
        im[(points[0]).astype(np.int), (points[1]).astype(np.int)] = 1
        im = ndimage.gaussian_filter(im, sigma=1280 / (4. * points.shape[1]))
        im = (im - im.min()) / (im.max() - im.min())

        return im.transpose()

    def generate_hms(self):
        hand_hms_dict = {}
        for index in self.data.index:
            if index % 100 == 0:
                print(f'index: {index}')
            # df_item = self.data.loc[index, :]
            img_file = self.data.loc[index, 'img_file']
            hand_bbox = self.data.loc[index, 'hand_bbox']
            # img = Image.open(img_file).convert('RGB')

            if hand_bbox is None:
                hand_hm = self.static_hm
            else:
                img = Image.open(img_file).convert('RGB')
                hand_hm = self.generate_hand_hm(img, hand_bbox)

            hand_hm = Image.fromarray(hand_hm)
            hand_hm = hand_hm.resize((self.args.img_resize[1],
                                      self.args.img_resize[0]))
            hand_hms_dict[img_file] = hand_hm

        save_file = open(os.path.join(self.data_path,
                                      f'{self.args.mode}_hand_hms.pkl'),
                         'wb')
        pickle.dump(hand_hms_dict, save_file)
        save_file.close()

    def generate_img_mask_pair(self):
        import shutil
        import cv2
        save_path = '/media/kaka/HD2T/dataset/ADL/saliency/val/'

        for index in self.data.index:
            if index % 500 == 0:
                print(f'index: {index}')

            img_file = self.data.loc[index, 'img_file']
            img = Image.open(img_file).convert('RGB')

            nao_bbox = self.data.loc[index, 'bboxes']

            mask = self.generate_mask(nao_bbox)

            shutil.copy(img_file, os.path.join(save_path,
                                               f'imgs/{str(index).zfill(6)}.jpg'))
            cv2.imwrite(os.path.join(save_path,
                                     f'gts/{str(index).zfill(6)}.jpg'),
                        255 * mask)


def show(img, mask):
    import matplotlib.pyplot as plt
    # for img in imgs:
    #     for idx in range(img.shape[0]):
    #         img_ = img[idx, :, :, :].cpu().numpy()
    #         mask_ = masks[0, idx, :, :].cpu().numpy()
    #         img_mask_ = (img_ * np.tile(mask_, (3, 1, 1))).transpose(1, 2, 0)
    #
    #         plt.imshow(img_mask_)
    #         plt.show()
    for idx in range(img.shape[0]):
        img_ = img[idx, :, :, :].cpu().numpy()
        mask_ = mask[idx, :, :].cpu().numpy()
        img_mask_ = (img_ * np.tile(mask_, (3, 1, 1))).transpose(1, 2, 0)
        plt.imshow(img_mask_)
        plt.show()


def plot_classes(data_df, feature, fs=8, show_percents=True,
                 color_palette='Set3'):
    import matplotlib.pyplot as plt
    import seaborn as sns
    f, ax = plt.subplots(1, 1, figsize=(2 * fs, 4))
    total = float(len(data_df))
    g = sns.countplot(data_df[feature],
                      order=data_df[feature].value_counts().index,
                      palette=color_palette)
    g.set_title(f"Number and percentage of labels for each class of {feature}")
    if show_percents:
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., height + 3,
                    '{:1.2f}%'.format(100 * height / total),
                    ha="center")
    plt.show()


def plot_seq_count(df, feature):
    import matplotlib.pyplot as plt
    # seq_len = df.bs_idx.value_counts().unique()
    seqs = df[feature].value_counts().value_counts().reset_index()
    seqs.columns = ['seq_len', 'seq_num']

    def draw_bar(x, y, x_label, y_label):
        fig = plt.figure(1)
        # plt.bar(obj_frame['num_object'].array, obj_frame['num_frame'].array)
        plt.bar(x, y)
        for xx, yy in zip(x, y):
            plt.text(xx, yy + 0.1, str(yy), ha='center')
        plt.xlabel(x_label)
        plt.ylabel(y_label)

        # plt.savefig("./asset/1.jpg")
        plt.show()

    draw_bar(seqs['seq_len'].array, seqs['seq_num'].array, 'seq_len', 'seq_num')


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # train_dataset = AdlSequenceDataset(args)
    # train_dataset = AdlDataset(args)
    args.mode = 'val'
    # train_dataset = AdlDatasetLabeled(args)
    # train_dataset = AdlSeq3Dataset(args)
    train_dataset = AdlDatasetV2(args)
    # train_dataset.generate_img_mask_pair()
    # train_dataset.generate_hms()
    # train_dataloader = DataLoader(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=1,
                                  num_workers=1, shuffle=True)
    sequence_lens = []
    for i, data in enumerate(train_dataloader):
        img, mask = data
        sequence_lens.append(img.shape[0])
        show(img, mask)
        print(img.shape)

"""
plt.bar(obj_frame['num_object'].array, obj_frame['num_frame'].array)
for xx, yy in zip(obj_frame['num_object'].array, obj_frame['num_frame'].array):
    plt.text(xx, yy + 0.1, str(yy), ha='center')
plt.xlabel(f'num of objects in single frame')
plt.ylabel(f'num of frames')
plt.savefig('./asset/adl_object_each_frame.svg', format='svg', dpi=1000)
plt.savefig('./asset/adl_object_each_frame.eps', format='eps', dpi=1000)

"""
