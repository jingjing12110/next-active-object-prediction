# @File : to_plot.py 
# @Time : 2019/10/30 
# @Email : jingjingjiang2017@gmail.com 

import os
import pickle
import numpy as np
import pandas as pd

import cv2
import seaborn as sns
import matplotlib.pyplot as plt

import torch.nn as nn
from data.adl import *
from model.unet_resnet import UNetResNet18
from model.unet_resnet_hand_att import UNetResnetHandAtt


test_softmax = nn.Softmax(dim=-1)


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

hm_transform = transforms.Compose([
    transforms.ToTensor()])


adl_data = AdlDatasetV2(args)
save_path = '/home/kaka/SSD_data/ADL/result'


def plot_cam():
    pass


def plot_ideal_output():
    # adl_df = adl_data.data
    adl_df = make_sequence_dataset_v2(args)

    video_id = 'P_01'
    img_name = '002869.jpg'
    img_file = f'/home/kaka/SSD_data/ADL/ADL_key_frames/{video_id}/{img_name}'
    bbox = adl_df[adl_df.img_file == img_file].bboxes


def plot_output():
    exp_name = 'adl/unet_resnet_hand_att_labeled'
    model = UNetResnetHandAtt()

    checkpoint = torch.load(os.path.join(args.exp_path, exp_name, 'ckpts/',
                                         f'model_epoch_{886}.pth'),
                            map_location='cpu')
    model.load_state_dict(checkpoint['net'])

    model.cpu()
    model.eval()

    video_id = 'P_01'
    img_name = '018793.jpg'
    img_file = f'/home/kaka/SSD_data/ADL/ADL_key_frames/{video_id}/{img_name}'
    img = Image.open(img_file).convert('RGB')
    img = img.resize((320, 224), Image.ANTIALIAS)
    img = img_transform(img)

    hand_hms = adl_data.hand_hms
    h_hm = np.array(hand_hms[img_file])
    h_hm = hm_transform(h_hm)

    outputs = model(img.unsqueeze(dim=0), h_hm.unsqueeze(dim=0))

    # compute softmax
    out = outputs.view(outputs.shape[0], outputs.shape[1], -1).permute(0, 2,
                                                                      1)
    out = test_softmax(out).permute(0, 2, 1)
    out = out.view(outputs.shape[0], outputs.shape[1], 224, -1)

    out1 = out.squeeze(dim=0).detach().numpy()[1, :, :]   # nao label=1
    # x, y = np.where(out1 < 0.2)
    # out1[x, y] = 0.2
    heatmap = cv2.applyColorMap(np.uint8(255 * out1), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    im = cv2.imread(img_file)
    im = cv2.resize(im, (320, 224)) / 255

    cam = heatmap * 0.3 + im * 0.5
    plt.imshow(cam, cmap='jet')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(save_path, f'th_cam_{video_id}_{img_name[:-4]}.png'),
                bbox_inches='tight', dpi=100, pad_inches=0.0)

    # use score threshold
    score_ = out.data.max(1)[0].cpu().numpy()
    score = score_.squeeze(0)
    # x, y = np.where(score < 0.8)
    # score[x, y] = 0.
    score = (score - score.min())/(score.max() - score.min())

    out_save = os.path.join(save_path, f'Out_{video_id}_{img_name[:-4]}')
    if not os.path.exists(out_save):
        os.mkdir(out_save)

    plt.imshow(score, cmap='jet')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(out_save, f'fm_{video_id}_{img_name[:-4]}.png'),
                bbox_inches='tight', dpi=100, pad_inches=0.0)
    plt.savefig(os.path.join(out_save, f'fm_{video_id}_{img_name[:-4]}.eps'),
                bbox_inches='tight', dpi=100, pad_inches=0.0)
    plt.show()

    x, y = np.where(score < 0.9)
    score[x, y] = 0.
    # inds = outputs.data.max(1)[1].cpu().numpy()
    # inds[x, y, z] = 0
    plt.imshow(score, cmap='jet')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(out_save, f'th_{video_id}_{img_name[:-4]}.png'),
                bbox_inches='tight', dpi=100, pad_inches=0.0)
    plt.savefig(os.path.join(out_save, f'th_{video_id}_{img_name[:-4]}.eps'),
                bbox_inches='tight', dpi=100, pad_inches=0.0)
    plt.show()


def plot_hand_prior_map():
    hand_hms = adl_data.hand_hms
    adl_df = adl_data.data
    # adl_df = make_sequence_dataset_v2(args)

    video_id = 'P_01'
    img_name = '002869.jpg'
    img_file = f'/home/kaka/SSD_data/ADL/ADL_key_frames/{video_id}/{img_name}'

    h_hm = np.array(hand_hms[img_file])

    hm_prior_save = os.path.join(save_path, f'Hand_Prior_{video_id}_{img_name[:-4]}')
    if not os.path.exists(hm_prior_save):
        os.mkdir(hm_prior_save)

    plt.imshow(h_hm, cmap='jet')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(os.path.join(hm_prior_save, f'hm_{video_id}_{img_name[:-4]}.png'),
                bbox_inches='tight', dpi=100, pad_inches=0.0)
    plt.savefig(os.path.join(hm_prior_save, f'hm_{video_id}_{img_name[:-4]}.eps'),
                bbox_inches='tight', dpi=100, pad_inches=0.0)
    plt.show()


def plot_hand_conv_map():
    exp_name = 'adl/unet_resnet_hand_att_labeled'
    model = UNetResnetHandAtt()
    # model = torch.nn.DataParallel(model)

    # model.load_state_dict(torch.load(os.path.join(args.exp_path, exp_name,
    #                                               'ckpts',
    #                                               f'model_epoch_{886}.pth'),
    #                                  map_location='cpu'))
    checkpoint = torch.load(os.path.join(args.exp_path, exp_name, 'ckpts/',
                                         f'model_epoch_{886}.pth'),
                            map_location='cpu')
    model.load_state_dict(checkpoint['net'])
    model.eval()

    hand_conv = model.conv_1x1

    video_id = 'P_01'
    img_name = '018793.jpg'
    img_file = f'/home/kaka/SSD_data/ADL/ADL_key_frames/{video_id}/{img_name}'

    hand_hms = adl_data.hand_hms
    h_hm = np.array(hand_hms[img_file])
    h_hm = hm_transform(h_hm)

    hm_conv = hand_conv(h_hm.unsqueeze(dim=0))

    hm_conv = hm_conv[0, :, :, :].permute(1, 2, 0).detach().numpy()

    hm_conv_save = os.path.join(save_path, f'Hand_Conv_{video_id}_{img_name[:-4]}')
    if not os.path.exists(hm_conv_save):
        os.mkdir(hm_conv_save)

    # l = 0
    for l in range(2):
        plt.imshow(hm_conv[:, :, l]/hm_conv.max(), cmap='jet')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(hm_conv_save, f'fm_{video_id}_{img_name[:-4]}_l{l}.png'),
                    bbox_inches='tight', dpi=100, pad_inches=0.0)
        plt.savefig(os.path.join(hm_conv_save, f'fm_{video_id}_{img_name[:-4]}_l{l}.eps'),
                    bbox_inches='tight', dpi=100, pad_inches=0.0)
        plt.show()


def gen_feature_map():
    exp_name = 'adl/unet_resnet_labeled'
    model = UNetResNet18()

    model.load_state_dict(torch.load(os.path.join(args.exp_path, exp_name,
                                                  'ckpts',
                                                  f'model_epoch_{1077}.pth'),
                                     map_location='cpu'))
    model.eval()

    # video_id = 'P_01'
    # img_name = '018793.jpg'
    # img_file = f'/home/kaka/SSD_data/ADL/ADL_key_frames/{video_id}/{img_name}'
    img_file = '/media/kaka/HD2T/P_14_Moment01.jpg'
    img = Image.open(img_file).convert('RGB')
    img = img.resize((320, 224), Image.ANTIALIAS)
    img = img_transform(img)

    out, features = model(img.unsqueeze(dim=0), with_output_feature_map=True)
    # feas = features.squeeze(dim=0).permute(1, 2, 0).detach().numpy()

    out1 = out.view(out.shape[0], out.shape[1], -1).permute(0, 2, 1)
    out1 = test_softmax(out1).permute(0, 2, 1)
    out1 = out1.view(out.shape[0], out.shape[1], 224, -1)
    # use score threshold
    # score_ = out1.data.max(1)[0].cpu().numpy()
    # score = score_.squeeze(0)
    out1 = out1.squeeze(dim=0).detach().cpu().numpy()[0, :, :]
    heatmap = cv2.applyColorMap(np.uint8(255 * out1), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    im = cv2.imread(img_file)
    im = cv2.resize(im, (320, 224)) / 255

    a = np.zeros((224, 320, 3))
    a[140:210, 130:210, :] = heatmap[140:210, 130:210, :]
    # a = heatmap
    x, y = np.where(a[:, :, 0] < 0.4)
    a[x, y, :] = 0


    cam = heatmap * 0.3 + im * 0.7
    # cam = a * 0.3 + im * 0.7
    cam = (cam - cam.min()) / (cam.max() - cam.min())

    plt.imshow(cam, cmap='jet')
    plt.xticks([])
    plt.yticks([])
    plt.savefig('./cam1.jpg', bbox_inches='tight', dpi=120, pad_inches=0.0)
    # plt.savefig(os.path.join(img_src, f'{img_file[-14:]}'),
    #             bbox_inches='tight', dpi=120, pad_inches=0.0)


    # plot_feas64(feas, video_id, img_name)

    # l = 0
    # plt.imshow(feas[:, :, l]/feas.max(), cmap='jet')
    # plt.xticks([])
    # plt.yticks([])
    # plt.savefig(os.path.join(save_path, f'fm_{video_id}_{img_name[:-4]}_l{l}.png'),
    #             bbox_inches='tight', dpi=100, pad_inches=0.0)
    # plt.savefig(os.path.join(save_path, f'fm_{video_id}_{img_name[:-4]}_l{l}.eps'),
    #             bbox_inches='tight', dpi=100, pad_inches=0.0)
    # plt.show()


def plot_feas64(feas, video_id, img_name):
    fea_save = os.path.join(save_path, f'Fea_{video_id}_{img_name[:-4]}')
    if not os.path.exists(fea_save):
        os.mkdir(fea_save)
    for l in range(64):
        plt.imshow(feas[:, :, l] / feas.max(), cmap='jet')
        plt.xticks([])
        plt.yticks([])
        plt.savefig(
            os.path.join(fea_save, f'fm_{video_id}_{img_name[:-4]}_l{l}.png'),
            bbox_inches='tight', dpi=100, pad_inches=0.0)
        # plt.savefig(
        #     os.path.join(fea_save, f'fm_{video_id}_{img_name[:-4]}_l{l}.eps'),
        #     bbox_inches='tight', dpi=100, pad_inches=0.0)


if __name__ == '__main__':
    # plot_hand_prior_map()
    gen_feature_map()
    # plot_hand_conv_map()
    # plot_ideal_output()
    # plot_output()

