# @File : opt.py
# @Time : 2019/7/30
# @Email : jingjingjiang2017@gmail.com

import argparse
import random

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--dataset', type=str, default='ADL',
                    help='EPIC or ADL')
parser.add_argument('--data_path',
                    # default='/home/kaka/SSD_data/ADL',
                    default='/media/kaka/HD2T/dataset/EPIC_KITCHENS/data/',
                    help='root path of ADL dataset')
parser.add_argument('--exp_path',
                    default='/media/kaka/HD2T/code/next_active_object/experiments/',
                    help='experiment path')
parser.add_argument('--exp_name', default='exp_name', type=str,
                    help='experiment path')

parser.add_argument('--img_size', default=[480, 640],  # [482, 642]
                    help='image size: [H, W]')  #
parser.add_argument('--img_resize', default=[224, 320],  # default=[242, 322]
                    help='image resize: [H, W]')  #
parser.add_argument('--normalize', default=True, help='subtract mean value')
parser.add_argument('--crop', default=False, help='')
parser.add_argument('--feature_len', type=int, default=6,
                    help='number of frames using for memory module')

parser.add_argument('--mode', default='train',
                    help='train , val or test')
parser.add_argument('--device_ids', nargs='+', default=[0], type=int)
parser.add_argument('--gpu_ids', default='0', type=str,
                    help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--debug', default=False, help='debug')
parser.add_argument('--print_every', type=int, default=10)

parser.add_argument('--bs', default=1, type=int, help='batch size')
parser.add_argument('--epochs', default=2000, type=int, help='Number of epochs')
parser.add_argument('--lr', default=0.0000002, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, help='momentum')
parser.add_argument('--weight_decay', default=0.0005, help='weight decay')

args = parser.parse_args()
print(args)
train_args = args.__dict__

if args.dataset == 'ADL':
    ids_adl = {'P_01', 'P_02', 'P_03', 'P_04', 'P_05', 'P_07', 'P_08',
               'P_09', 'P_10', 'P_11', 'P_12', 'P_13', 'P_14', 'P_18',
               'P_06', 'P_15', 'P_16', 'P_17', 'P_19', 'P_20'}

    val_video_id = ['P_05', 'P_04', 'P_07', 'P_10', 'P_16']
    train_video_id = ids_adl - set(val_video_id)
    test_video_id = val_video_id

    args.data_path = '/home/kaka/SSD_data/ADL'
    annos_path = 'ADL_annotations/object_annotation'
    annos_path_v2 = 'ADL_annotations/nao_annotation'
    frames_path = 'ADL_key_frames'  # 'ADL_frames'
    features_path = 'ADL_key_features'

    if args.debug:
        # train_video_id = {'P_01', 'P_02', 'P_03', 'P_04', 'P_05', 'P_06'}
        train_video_id = ['P_02']
        val_video_id = ['P_18']
        
    action_name = {
        'empty': 0,
        'combing hair': 1,
        'make up': 2,
        'brushing teeth': 3,
        'dental floss': 4,
        'washing hands/face': 5,
        'drying hands/face': 6,
        'enter/leave room': 7,
        'adjusting thermostat': 8,
        'laundry': 9,
        'washing dishes': 10,
        'moving dishes': 11,
        'making tea': 12,
        'making coffee': 13,
        'drinking water/bottle': 14,
        'drinking water/tap': 15,
        'making hot food': 16,
        'making cold food/snack': 17,
        'eating food/snack': 18,
        'mopping in kitchen': 19,
        'vacuuming': 20,
        'taking pills': 21,
        'watching tv': 22,
        'using computer': 23,
        'using cell': 24,
        'making bed': 25,
        'cleaning house': 26,
        'reading book': 27,
        'using_mouth_wash': 28,
        'writing': 29,
        'putting on shoes/sucks': 30,
        'drinking coffee/tea': 31,
        'grabbing water from tap': 32
    }

else:
    id = {'P01P01_01', 'P01P01_02', 'P01P01_03', 'P01P01_04', 'P01P01_05',
          'P01P01_06', 'P01P01_07', 'P01P01_08', 'P01P01_09', 'P01P01_10',
          'P01P01_16', 'P01P01_17', 'P01P01_18', 'P01P01_19', 'P02P02_01',
          'P02P02_02', 'P02P02_03', 'P02P02_04', 'P02P02_05', 'P02P02_06',
          'P02P02_07', 'P02P02_08', 'P02P02_09', 'P02P02_10', 'P02P02_11',
          'P03P03_02', 'P03P03_03', 'P03P03_04', 'P03P03_05', 'P03P03_06',
          'P03P03_07', 'P03P03_08', 'P03P03_09', 'P03P03_10', 'P03P03_11',
          'P03P03_12', 'P03P03_13', 'P03P03_14', 'P03P03_15', 'P03P03_16',
          'P03P03_17', 'P03P03_18', 'P03P03_19', 'P03P03_20', 'P03P03_27',
          'P03P03_28', 'P04P04_01', 'P04P04_02', 'P04P04_03', 'P04P04_04',
          'P04P04_05', 'P04P04_06', 'P04P04_07', 'P04P04_08', 'P04P04_09',
          'P04P04_10', 'P04P04_11', 'P04P04_12', 'P04P04_13', 'P04P04_14',
          'P04P04_15', 'P04P04_16', 'P04P04_17', 'P04P04_18', 'P04P04_19',
          'P04P04_20', 'P04P04_21', 'P04P04_22', 'P04P04_23', 'P05P05_01',
          'P05P05_02', 'P05P05_03', 'P05P05_04', 'P05P05_05', 'P05P05_06',
          'P05P05_08', 'P06P06_01', 'P06P06_02', 'P06P06_03', 'P06P06_05',
          'P06P06_07', 'P06P06_08', 'P06P06_09', 'P07P07_01', 'P07P07_02',
          'P07P07_03', 'P07P07_04', 'P07P07_05', 'P07P07_06', 'P07P07_07',
          'P07P07_08', 'P07P07_09', 'P07P07_10', 'P07P07_11', 'P08P08_01',
          'P08P08_02', 'P08P08_03', 'P08P08_04', 'P08P08_05', 'P08P08_06',
          'P08P08_07', 'P08P08_08', 'P08P08_11', 'P08P08_12', 'P08P08_13',
          'P08P08_18', 'P08P08_19', 'P08P08_20', 'P08P08_21', 'P08P08_22',
          'P08P08_23', 'P08P08_24', 'P08P08_25', 'P08P08_26', 'P08P08_27',
          'P08P08_28', 'P10P10_01', 'P10P10_02', 'P10P10_04', 'P12P12_01',
          'P12P12_02', 'P12P12_04', 'P12P12_05', 'P12P12_06', 'P12P12_07',
          'P13P13_04', 'P13P13_05', 'P13P13_06', 'P13P13_07', 'P13P13_08',
          'P13P13_09', 'P13P13_10', 'P14P14_01', 'P14P14_02', 'P14P14_03',
          'P14P14_04', 'P14P14_05', 'P14P14_07', 'P14P14_09', 'P15P15_01',
          'P15P15_02', 'P15P15_03', 'P15P15_07', 'P15P15_08', 'P15P15_09',
          'P15P15_10', 'P15P15_11', 'P15P15_12', 'P15P15_13', 'P16P16_01',
          'P16P16_02', 'P16P16_03', 'P17P17_01', 'P17P17_03', 'P17P17_04',
          'P19P19_01', 'P19P19_02', 'P19P19_03', 'P19P19_04', 'P20P20_01',
          'P20P20_02', 'P20P20_03', 'P20P20_04', 'P21P21_01', 'P21P21_03',
          'P21P21_04', 'P22P22_05', 'P22P22_06', 'P22P22_07', 'P22P22_08',
          'P22P22_09', 'P22P22_10', 'P22P22_11', 'P22P22_12', 'P22P22_13',
          'P22P22_14', 'P22P22_15', 'P22P22_16', 'P22P22_17', 'P23P23_01',
          'P23P23_02', 'P23P23_03', 'P23P23_04', 'P24P24_01', 'P24P24_02',
          'P24P24_03', 'P24P24_04', 'P24P24_05', 'P24P24_06', 'P24P24_07',
          'P24P24_08', 'P25P25_01', 'P25P25_02', 'P25P25_03', 'P25P25_04',
          'P25P25_05', 'P25P25_09', 'P25P25_10', 'P25P25_11', 'P25P25_12',
          'P26P26_01', 'P26P26_02', 'P26P26_03', 'P26P26_04', 'P26P26_05',
          'P26P26_06', 'P26P26_07', 'P26P26_08', 'P26P26_09', 'P26P26_10',
          'P26P26_11', 'P26P26_12', 'P26P26_13', 'P26P26_14', 'P26P26_15',
          'P26P26_16', 'P26P26_17', 'P26P26_18', 'P26P26_19', 'P26P26_20',
          'P26P26_21', 'P26P26_22', 'P26P26_23', 'P26P26_24', 'P26P26_25',
          'P26P26_26', 'P26P26_27', 'P26P26_28', 'P26P26_29', 'P27P27_01',
          'P27P27_02', 'P27P27_03', 'P27P27_04', 'P27P27_06', 'P27P27_07',
          'P28P28_01', 'P28P28_02', 'P28P28_03', 'P28P28_04', 'P28P28_05',
          'P28P28_06', 'P28P28_07', 'P28P28_08', 'P28P28_09', 'P28P28_10',
          'P28P28_11', 'P28P28_12', 'P28P28_13', 'P28P28_14', 'P29P29_01',
          'P29P29_02', 'P29P29_03', 'P29P29_04', 'P30P30_01', 'P30P30_02',
          'P30P30_03', 'P30P30_04', 'P30P30_05', 'P30P30_06', 'P30P30_10',
          'P30P30_11', 'P31P31_01', 'P31P31_02', 'P31P31_03', 'P31P31_04',
          'P31P31_05', 'P31P31_06', 'P31P31_07', 'P31P31_08', 'P31P31_09',
          'P31P31_13', 'P31P31_14'}

    test_video_id = \
        {'P08P08_11', 'P16P16_03', 'P29P29_04', 'P22P22_05', 'P31P31_04',
         'P07P07_01', 'P19P19_02', 'P25P25_09', 'P25P25_12', 'P26P26_26',
         'P02P02_06', 'P07P07_09', 'P08P08_21', 'P28P28_02', 'P04P04_18',
         'P20P20_03', 'P08P08_19', 'P01P01_17', 'P24P24_01', 'P13P13_04',
         'P28P28_14', 'P12P12_07', 'P19P19_01', 'P13P13_05', 'P04P04_12',
         'P26P26_08', 'P13P13_08', 'P05P05_01', 'P31P31_14', 'P26P26_27',
         'P15P15_10', 'P24P24_05', 'P15P15_13', 'P31P31_09', 'P04P04_17',
         'P08P08_04', 'P03P03_05', 'P05P05_04', 'P28P28_10', 'P22P22_12',
         'P03P03_15', 'P24P24_07', 'P25P25_03', 'P03P03_09', 'P15P15_12',
         'P08P08_07', 'P04P04_11', 'P20P20_04', 'P26P26_05', 'P07P07_05',
         'P25P25_01', 'P04P04_04', 'P30P30_02', 'P26P26_20', 'P10P10_02',
         'P04P04_06', 'P07P07_02', 'P29P29_01', 'P06P06_02', 'P08P08_05',
         'P05P05_08', 'P07P07_06', 'P26P26_06', 'P31P31_06', 'P13P13_07',
         'P21P21_03', 'P20P20_01', 'P08P08_01', 'P12P12_05', 'P22P22_15',
         'P26P26_03', 'P06P06_09'}
    train_video_id = id - test_video_id

    val_video_id = test_video_id
    annos_path = 'nao_annotations'
    frames_path = 'object_detection_images/train'  #
    features_path = 'EPIC_key_features'

    if args.debug:
        # train_video_id = {'P_01', 'P_02', 'P_03', 'P_04', 'P_05', 'P_06'}
        train_video_id = ['P01P01_01']
        val_video_id = ['P31P31_14']
