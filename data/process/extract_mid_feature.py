import math
import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms

# from data_loader.adl import AdlSequenceDataset
# from data.adl import make_sequence_dataset
from data.epic import make_sequence_dataset
from opt import args

# 间隔采样的长度
SEQUENCE_LEN = 6  # 3, 5, 6, 9


class FeatureExtraction(object):
    def __init__(self, seq_len, use_gpu=True, use_model='vgg16',
                 use_data='train'):
        self.SEQUENCE_LEN = seq_len
        self.use_gpu = use_gpu
        self.use_model = use_model
        self.use_data = use_data

        self.data = make_sequence_dataset(args)
        self.adl_imgs_path = '/media/kaka/HD2T/dataset/EPIC_KITCHENS/data/object_detection_images/train'
        self.save_feature_path = '/media/kaka/HD2T/dataset/EPIC_KITCHENS/data'

        self.data = pd.read_csv(os.path.join(self.save_feature_path,
                                             f'{use_data}_seq_df.csv'))

        self.TRANSFORM = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        if use_model == 'vgg16':
            self.model = models.vgg16(pretrained=True).features
        elif use_model == 'resnet18':
            self.model = nn.Sequential(
                *list(models.resnet18(pretrained=True).children())[:-2])

        if use_gpu:
            self.model = self.model.cuda()

    def visulize_feature(self, features):
        for i in range(features.shape[1]):
            feature = features[:, i, :, :]
            feature = feature.view(feature.shape[1], feature.shape[2])
            feature = feature.data.numpy()
            # use sigmod to [0,1]
            feature = 1.0 / (1 + np.exp(-1 * feature))
            # to [0,255]
            feature = np.round(feature * 255)

    def extract_feature(self):
        features_dict = {}
        for idx in range(self.data.bs_idx.unique().shape[0]):
            # for idx in [70]:
            print(f'start: {idx} !')
            df_item = self.data[self.data['bs_idx'] == idx]
            df_item = df_item.sort_values(by='img_file').reset_index(drop=True)

            if df_item.shape[0] == 3:
                # 0-1 ############################################################
                img_file_0 = df_item.iloc[0, 0]
                frame_0 = int(img_file_0[-10:-4])
                frame_1 = int(df_item.iloc[1, 0][-10:-4])  # img_file_1
                frames = np.linspace(frame_0, frame_1, self.SEQUENCE_LEN).astype(
                    np.int64)

                # 提取选择的中间帧的feature
                features1 = np.zeros([self.SEQUENCE_LEN, 512, 70])
                for i, frame_ in enumerate(sorted(frames)):
                    img_path = os.path.join(self.adl_imgs_path,
                                            img_file_0[-15:-10],
                                            f'{str(frame_).zfill(6)}.jpg')
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((320, 224), Image.ANTIALIAS)

                    img = self.TRANSFORM(img)
                    # img = img.view([1, 3, 224, 320])
                    img = Variable(torch.unsqueeze(img, dim=0).float(),
                                   requires_grad=False)
                    if self.use_gpu:
                        img = img.cuda()
                    feature = self.model(img)
                    feature = feature[0, :, :, :].view(512,
                                                       -1).detach().cpu().numpy()

                    features1[i, :, :] = feature

                # 1-2 ############################################################
                frame_2 = int(df_item.iloc[2, 0][-10:-4])  # img_file_2

                features2 = np.zeros([self.SEQUENCE_LEN, 512, 70])
                if frame_2 == frame_1:
                    features2 = features1
                else:
                    frames = np.linspace(frame_1, frame_2, SEQUENCE_LEN).astype(
                        np.int64)
                    for i, frame_ in enumerate(sorted(frames)):
                        img_path = os.path.join(self.adl_imgs_path,
                                                img_file_0[-15:-10],
                                                f'{str(frame_).zfill(6)}.jpg')
                        img = Image.open(img_path).convert('RGB')
                        img = img.resize((320, 224), Image.ANTIALIAS)

                        img = self.TRANSFORM(img)
                        img = Variable(torch.unsqueeze(img, dim=0).float(),
                                       requires_grad=False)
                        if self.use_gpu:
                            img = img.cuda()
                        feature = self.model(img)
                        feature = feature[0, :, :, :].view(512,
                                                           -1).detach().cpu().numpy()

                        features2[i, :, :] = feature

                features_dict[f'{idx}'] = [features1, features2]
            else:
                print(f'len= {df_item.shape[0]}')

        save_file = open(os.path.join(self.save_feature_path,
                                      f'{self.use_data}_{self.use_model}_features.pickle'),
                         'wb')
        pickle.dump(features_dict, save_file)
        save_file.close()
        # feas_file = open('train_feature_bs_idx.pickle', 'rb')
        # feas = pickle.load(feas_file)

    def cut_seq_df(self):
        # data_df = pd.DataFrame(columns=['img_file', 'bboxes', 'track_id',
        #                                 'label', 'bs_idx'])
        data_df = pd.DataFrame(columns=['img_file', 'pseudo_track_id', 'nao_bbox',
                                        'label', 'bs_idx'])
        for i in range(self.data.bs_idx.unique().shape[0]):
            df_item = self.data[self.data['bs_idx'] == str(i)]

            if df_item.shape[0] == 2:
                df_item = df_item.append(df_item[-1:])
            elif (df_item.shape[0] >= 3) & (df_item.shape[0] <= 5):
                df_item = df_item[-3:]
            else:
                df_item = df_item.iloc[[0, math.ceil(df_item.shape[0] / 3),
                                        df_item.shape[0] - 1]]
            df_item = df_item.reset_index(drop=True)
            data_df = pd.concat([data_df, df_item], ignore_index=True)

        data_df.to_csv(os.path.join(self.save_feature_path,
                                    f'{self.use_data}_seq_df.csv'), index=False)


# def main(args):
#     adl_dataset = AdlSequenceDataset(args)
#     data_df = adl_dataset.data
#
#     adl_dataloader = DataLoader(adl_dataset, batch_size=4,
#                                 num_workers=3, shuffle=False)
#
#     vgg16 = models.vgg16(pretrained=True).features
#     modules = list(models.resnet18(pretrained=True).children())[:-2]
#     resnet18 = nn.Sequential(*modules)
#
#     for i, data in enumerate(adl_dataloader):
#         pass


if __name__ == '__main__':
    # main(args)
    FeaExt = FeatureExtraction(seq_len=6)
    FeaExt.extract_feature()
