# @File : annote_active_object_v2.py.py 
# @Time : 2019/9/16 
# @Email : jingjingjiang2017@gmail.com

import os
from ast import literal_eval

import cv2
# import numpy as np
import pandas as pd


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    import random
    import colorsys
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


class EpicDataParsing(object):
    def __init__(self):
        super(EpicDataParsing, self).__init__()

        self.data_path = '/media/kaka/HD2T/dataset/EPIC_KITCHENS/data/'
        self.img_path = self.data_path + 'object_detection_images/train'
        self.anno_path = self.data_path + 'annotations'
        self.NAO_path = self.data_path + 'active_annotations'

        self.annos = pd.read_csv(os.path.join(self.anno_path,
                                              'EPIC_train_object_labels.csv'),
                                 converters={"bounding_boxes": literal_eval})
        # 去掉df中bounding box=[]
        self.annos = self.annos[self.annos.astype(str)['bounding_boxes'] != '[]']
        self.annos.loc[:, 'id'] = self.annos['participant_id'] + self.annos[
            'video_id'] 

        self.box_scale = 80
        self.labeled_ids = ['P01P01_01', 'P01P01_02', 'P01P01_03', 'P01P01_04',
                            'P01P01_05', 'P01P01_06', 'P01P01_07', 'P01P01_08',
                            'P01P01_09', 'P01P01_10', 'P01P01_16', 'P01P01_17',
                            'P01P01_18', 'P01P01_19', 'P02P02_01', 'P02P02_02',
                            'P02P02_03', 'P02P02_04', 'P02P02_05', 'P02P02_06',
                            'P02P02_07', 'P02P02_08', 'P02P02_09', 'P02P02_10',
                            'P02P02_11', 'P03P03_02', 'P03P03_03', 'P03P03_04',
                            'P03P03_05', 'P03P03_06', 'P03P03_07', 'P03P03_08',
                            'P03P03_09', 'P03P03_10', 'P03P03_11', 'P03P03_12',
                            'P03P03_13', 'P03P03_14', 'P03P03_15', 'P03P03_16',
                            'P03P03_17', 'P03P03_18', 'P03P03_19', 'P03P03_20',
                            'P03P03_27', 'P03P03_28', 'P04P04_01', 'P04P04_02',
                            'P04P04_03', 'P04P04_04', 'P04P04_05', 'P04P04_05',
                            'P04P04_06', 'P04P04_07', 'P04P04_08', 'P04P04_09',
                            'P04P04_10', 'P04P04_11', 'P04P04_12', 'P04P04_13',
                            'P04P04_14', 'P04P04_15', 'P04P04_16', 'P04P04_17',
                            'P04P04_18', 'P04P04_19', 'P04P04_20', 'P04P04_21',
                            'P04P04_22', 'P04P04_23', 'P05P05_01', 'P05P05_02',
                            'P05P05_03', 'P05P05_04', 'P05P05_05', 'P05P05_06',
                            'P05P05_08', 'P06P06_01', 'P06P06_02', 'P06P06_03',
                            'P06P06_05', 'P06P06_07', 'P06P06_08', 'P06P06_09',
                            'P07P07_01', 'P07P07_02', 'P07P07_03', 'P07P07_04',
                            'P07P07_05', 'P07P07_06', 'P07P07_07', 'P07P07_08',
                            'P07P07_09', 'P07P07_10', 'P07P07_11', 'P08P08_01',
                            'P08P08_02', 'P08P08_03', 'P08P08_04', 'P08P08_05',
                            'P08P08_06', 'P08P08_07', 'P08P08_08', 'P08P08_11',
                            'P08P08_12', 'P08P08_13', 'P08P08_18', 'P08P08_19',
                            'P08P08_20', 'P08P08_21', 'P08P08_22', 'P08P08_23',
                            'P08P08_24', 'P08P08_25', 'P08P08_26', 'P08P08_27',
                            'P08P08_28', 'P10P10_01', 'P10P10_02', 'P10P10_04',
                            'P12P12_01', 'P12P12_02', 'P12P12_04', 'P12P12_05',
                            'P12P12_06', 'P12P12_07', 'P13P13_04', 'P13P13_05',
                            'P13P13_06', 'P13P13_07', 'P13P13_08', 'P13P13_09',
                            'P13P13_10', 'P14P14_01', 'P14P14_02', 'P14P14_03',
                            'P14P14_04', 'P14P14_05', 'P14P14_07', 'P14P14_09',
                            'P15P15_01', 'P15P15_02', 'P15P15_03', 'P15P15_07',
                            'P15P15_08', 'P15P15_09', 'P15P15_10', 'P15P15_11',
                            'P15P15_12', 'P15P15_13', 'P16P16_01', 'P16P16_02',
                            'P16P16_03', 'P17P17_01', 'P17P17_03', 'P17P17_04',
                            'P19P19_01', 'P19P19_02', 'P19P19_03', 'P19P19_04',
                            'P20P20_01', 'P20P20_02', 'P20P20_03', 'P20P20_04',
                            'P21P21_01', 'P21P21_03', 'P21P21_04', 'P22P22_05',
                            'P22P22_06', 'P22P22_07', 'P22P22_08', 'P22P22_09',
                            'P22P22_10', 'P22P22_11', 'P22P22_12', 'P22P22_13',
                            'P22P22_14', 'P22P22_15', 'P22P22_16', 'P22P22_17',
                            'P23P23_01', 'P23P23_02', 'P23P23_03', 'P23P23_04',
                            'P24P24_01', 'P24P24_02', 'P24P24_03', 'P24P24_04',
                            'P24P24_05', 'P24P24_06', 'P24P24_07', 'P24P24_08',
                            'P25P25_01', 'P25P25_02', 'P25P25_03', 'P25P25_04',
                            'P25P25_05', 'P25P25_09', 'P25P25_10', 'P25P25_11',
                            'P25P25_12', 'P26P26_01', 'P26P26_02', 'P26P26_03',
                            'P26P26_04', 'P26P26_05', 'P26P26_06', 'P26P26_07',
                            'P26P26_08', 'P26P26_09', 'P26P26_10', 'P26P26_11',
                            'P26P26_12', 'P26P26_13', 'P26P26_14', 'P26P26_15',
                            'P26P26_16', 'P26P26_17', 'P26P26_18', 'P26P26_19',
                            'P26P26_20', 'P26P26_21', 'P26P26_22', 'P26P26_23',
                            'P26P26_24', 'P26P26_25', 'P26P26_26', 'P26P26_27',
                            'P26P26_28', 'P26P26_29', 'P27P27_01', 'P27P27_02',
                            'P27P27_03', 'P27P27_04', 'P27P27_06', 'P27P27_07',
                            'P28P28_01', 'P28P28_02', 'P28P28_03', 'P28P28_04',
                            'P28P28_05', 'P28P28_06', 'P28P28_07', 'P28P28_08',
                            'P28P28_09', 'P28P28_10', 'P28P28_11', 'P28P28_12',
                            'P28P28_13', 'P28P28_14', 'P29P29_01', 'P29P29_02',
                            'P29P29_03', 'P29P29_04', 'P30P30_01', 'P30P30_02',
                            'P30P30_03', 'P30P30_04', 'P30P30_05', 'P30P30_06',
                            'P30P30_10', 'P30P30_11', 'P31P31_01', 'P31P31_02',
                            'P31P31_03', 'P31P31_04', 'P31P31_05', 'P31P31_06',
                            'P31P31_07', 'P31P31_08', 'P31P31_09', 'P31P31_13',
                            'P31P31_14']
                      
    def data_annotation(self):
        for id in sorted(self.annos.id.unique()):  # participant_id + video_id
            if id not in self.labeled_ids:
                print(f'id: {id}')
                video_path = os.path.join(self.NAO_path, id)
                if not os.path.exists(video_path):
                    os.mkdir(video_path)

                df = self.annos[self.annos['id'] == id]

                print(f'video {id} has {df.frame.unique().shape[0]} frames.')

                self.data_annotation_video(df)

                self.labeled_ids.append(id)

                pd.DataFrame(self.labeled_ids).to_csv(os.path.join(self.data_path,
                                                                   'ids.txt'),
                                                      header=0)

    def data_annotation_video(self, df):
        """标注单个视频"""
        img_src = os.path.join(self.img_path, df.iloc[0, 2], df.iloc[0, 3])
        frames = sorted(df.frame.unique())
        print(f'frame max is : {max(frames)}')
        # open bbox annotation
        cv2.namedWindow("Frame", 0)
        # for idx in range(len(frames)):
        idx = 0
        while idx < len(frames):
            print('=============================================')
            frame = frames[idx]
            # print(f'frame= {frame:05d}')
            image = cv2.imread(os.path.join(img_src,
                                            f'{str(frame).zfill(10)}.jpg'))
            # 求每帧bbox和画图
            ann_objects, image, pid = self.df2objects(df, frame, image)

            cv2.imshow('Frame', image)
            flag_ = cv2.waitKey(0)

            if flag_ == 117:  # up
                print(f'key value in ASCII is: {flag_}/ up.')
                idx = idx - 1
                frame = frames[idx]
                image = cv2.imread(os.path.join(img_src,
                                                f'{str(frame).zfill(10)}.jpg'))
                ann_objects, image, pid = self.df2objects(df, frame, image)
                cv2.imshow('Frame', image)
                print(f'frame= {frame:010d}')
                # self.active_object_annotation(frame, image, ann_objects, pid)

            elif flag_ == 110:  # n
                print(f'key value in ASCII is: {flag_} /next.')
                idx = idx + 1
                frame = frames[idx]
                image = cv2.imread(os.path.join(img_src,
                                                f'{str(frame).zfill(10)}.jpg'))
                ann_objects, image, pid = self.df2objects(df, frame, image)
                cv2.imshow('Frame', image)
                print(f'frame= {frame:010d}')
                # self.active_object_annotation(frame, image, ann_objects, pid)

            elif flag_ == 99:  # c
                print(f'key value in ASCII is: {flag_} / current.')
                print(f'frame= {frame:010d}')
                print('labelling!!!!!!!!!!!!!!!!!!!!!!!!!')
                self.active_object_annotation(frame, image, ann_objects, pid)

                idx = idx + 1

        cv2.destroyAllWindows()

    def df2objects(self, df, frame, image):
        df_f = df[df['frame'] == frame]
        colors = random_colors(10)
        ann_objects = []
        object_names = []

        for index in sorted(df_f.index):
            for box in df_f.loc[index, 'bounding_boxes']:
                temp_ = [int(box[1]), int(box[0]),
                         int(box[1] + box[3]), int(box[0] + box[2])]
                ann_objects.append(temp_)
                object_names.append(df_f.loc[index, 'noun'])

        box_num = len(ann_objects)

        # 画bbox对应的点击框
        for j in range(box_num):
            box = ann_objects[j]
            b_lt = (box[0], box[1])
            b_rb = (box[2], box[3])
            color = colors[j]
            color255 = [i * 255 for i in color]
            cv2.rectangle(image, b_lt, b_rb, color255, 2)
            cv2.putText(image, f'{object_names[j]}/{frame}',
                        (box[0], int(box[1] + 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color255, 2)
            # this is novel that we draw a box with same color
            # with bbox to index it
            y1 = self.box_scale * j
            y2 = self.box_scale * (j + 1)
            x1 = 0
            x2 = 200
            # alpha = 0.5
            for c in range(3):
                image[y1:y2, x1:x2, c] = color255[c]

        # 画上/下帧的框
        # color_ = [[0, 0, 255], [0, 255, 0]]
        # for k in range(2):
        #     y1 = 1000
        #     y2 = 1080
        #     x1 = 200 * k
        #     x2 = 200 * (k + 1)
        #     for c in range(3):
        #         image[y1:y2, x1:x2, c] = color_[k][c]

        return ann_objects, image, df_f.iloc[0, 6]

    def active_object_annotation(self, frame, image, ann_objects, pid):
        """标注单帧"""
        data = []

        # mouse case, to define the clicked objects
        def on_mouse(event, x, y, flags, pos):
            if event == cv2.EVENT_LBUTTONDBLCLK:  # 双击鼠标左键
                data.append(x)
                data.append(y)

                if y > self.box_scale * 5:
                    for i in range(10):
                        print('error of pressing')

        jump = 0
        global_txt_write = {}  # save the previous info

        # jump ==0 means i will not skip, so waitkey(0) to allow
        # the choose of attention number as well as mouse click
        att_box_num = None
        if jump == 0:
            # cv2.imshow('Frame', image)
            cv2.setMouseCallback('Frame', on_mouse)
            flag = cv2.waitKey(0)  # this line is novel
            if flag == 49:  # 1
                # att_box_num.append('one')
                att_box_num = 'one'
                print('you have pressed 1')
            elif flag == 50:  # 2
                # att_box_num.append('two')
                att_box_num = 'two'
                print('you have pressed 2')
            elif flag == 51:  # 3
                att_box_num = 'three'
                print('you have pressed 3')
                # jump = 1
            elif flag == 32:  # 空格
                print('no box was chosen')

        # write annotation
        if att_box_num == 'one':
            if jump == 0:
                y = data[-1]
                att_box_idx = int(y / self.box_scale)
                bbox = ann_objects[att_box_idx]

                global_txt_write['num'] = 1
                global_txt_write['obj_idx'] = att_box_idx
            else:
                bbox = ann_objects[global_txt_write['obj_idx']]

            bbox_active_write = f'{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n'
            # print(bbox_active_write)
            with open(os.path.join(self.NAO_path,
                                   pid, f'{str(frame).zfill(10)}.txt'),
                      'a+') as f_txt:
                f_txt.write(bbox_active_write)

            # highlight the chose one
            # b_lt = (int(float(bbox[0])), int(float(bbox[1])))
            # b_rb = (int(float(bbox[2] + 5)), int(float(bbox[3] + 5)))
            # cv2.rectangle(image, b_lt, b_rb, (0, 0, 255), 2)

        if att_box_num == 'two':
            if jump == 0:
                # x1 = data[-2]
                y1 = data[-1]
                # x2 = data[-4]
                y2 = data[-3]

                att_box_idx_01 = int(y1 / self.box_scale)
                att_box_idx_02 = int(y2 / self.box_scale)
                bbox_01 = ann_objects[att_box_idx_01]
                bbox_02 = ann_objects[att_box_idx_02]

                global_txt_write['num'] = 2
                global_txt_write['obj_idx'] = [att_box_idx_01,
                                               att_box_idx_02]

            else:
                bbox_01 = ann_objects[global_txt_write['obj_idx'][0]]
                bbox_02 = ann_objects[global_txt_write['obj_idx'][1]]

            bbox_pos1 = f'{bbox_01[0]} {bbox_01[1]} {bbox_01[2]} {bbox_01[3]}\n'
            bbox_pos2 = f'{bbox_02[0]} {bbox_02[1]} {bbox_02[2]} {bbox_02[3]}\n'

            bbox_active_write = bbox_pos1 + bbox_pos2
            # print(bbox_active_write)
            with open(os.path.join(self.NAO_path,
                                   pid, f'{str(frame).zfill(10)}.txt'),
                      'a') as f_txt:
                f_txt.write(bbox_active_write)

            # highlight the chose two
            # b_lt = (bbox_01[0], bbox_01[1])
            # b_rb = (bbox_01[2] + 5, bbox_01[3] + 5)
            # cv2.rectangle(image, b_lt, b_rb, (0, 0, 255), 2)
            # b_lt = (bbox_02[0], bbox_02[1])
            # b_rb = (bbox_02[2] + 5, bbox_02[3] + 5)
            # cv2.rectangle(image, b_lt, b_rb, (0, 0, 255), 2)

        if att_box_num == 'three':
            if jump == 0:
                # x1 = data[-2]
                y1 = data[-1]
                # x2 = data[-4]
                y2 = data[-3]
                # x3 = data[-6]
                y3 = data[-5]

                att_box_idx_01 = int(y1 / self.box_scale)
                att_box_idx_02 = int(y2 / self.box_scale)
                att_box_idx_03 = int(y3 / self.box_scale)
                bbox_01 = ann_objects[att_box_idx_01]
                bbox_02 = ann_objects[att_box_idx_02]
                bbox_03 = ann_objects[att_box_idx_03]

                global_txt_write['num'] = 3
                global_txt_write['obj_idx'] = [att_box_idx_01,
                                               att_box_idx_02,
                                               att_box_idx_03]

            else:
                bbox_01 = ann_objects[global_txt_write['obj_idx'][0]]
                bbox_02 = ann_objects[global_txt_write['obj_idx'][1]]
                bbox_03 = ann_objects[global_txt_write['obj_idx'][2]]

            bbox_pos1 = f'{bbox_01[0]} {bbox_01[1]} {bbox_01[2]} {bbox_01[3]}\n'
            bbox_pos2 = f'{bbox_02[0]} {bbox_02[1]} {bbox_02[2]} {bbox_02[3]}\n'
            bbox_pos3 = f'{bbox_03[0]} {bbox_03[1]} {bbox_03[2]} {bbox_03[3]}\n'

            bbox_active_write = bbox_pos1 + bbox_pos2 + bbox_pos3
            # print(bbox_active_write)
            with open(os.path.join(self.NAO_path,
                                   pid, f'{str(frame).zfill(10)}.txt'),
                      'w') as f_txt:
                f_txt.write(bbox_active_write)

            # highlight the chose two
            # b_lt = (bbox_01[0], bbox_01[1])
            # b_rb = (bbox_01[2] + 5, bbox_01[3] + 5)
            # cv2.rectangle(image, b_lt, b_rb, (0, 0, 255), 2)
            # b_lt = (bbox_02[0], bbox_02[1])
            # b_rb = (bbox_02[2] + 5, bbox_02[3] + 5)
            # cv2.rectangle(image, b_lt, b_rb, (0, 0, 255), 2)
            # b_lt = (bbox_03[0], bbox_03[1])
            # b_rb = (bbox_03[2] + 5, bbox_03[3] + 5)
            # cv2.rectangle(image, b_lt, b_rb, (0, 0, 255), 2)

    def label_transform(self, id, is_show=True):
        label_path = os.path.join(self.NAO_path, id)
        img_path = os.path.join(self.data_path,
                                'object_detection_images/train_bbox',
                                str(id)[:3], str(id)[3:])
        annos = self.annos[self.annos['id'] == id]  # 选择视频id的label数据
        annos.insert(loc=7, column='has_nao', value=0)

        nao_frames = os.listdir(label_path)  # 读取nao的label
        nao_frames = [int(str(f).split('.')[0]) for f in nao_frames]

        print(f'{id} {annos.shape[0]} {len(nao_frames)}')

        nao_dict = {'frame': [], 'id': [], 'label': [], 'nao_bbox': []}
        frames = []
        ids = []
        labels = []
        nao_boxes = []

        for frame in sorted(nao_frames):
            anno_f = annos[annos['frame'] == frame]
            if id == 'P01P01_01':
                nao_bboxes = pd.read_csv(os.path.join(label_path,
                                                      f'{str(frame)}.txt'),
                                         header=None, delim_whitespace=True,
                                         names=['x1', 'y1', 'x2', 'y2'])
            else:
                nao_bboxes = pd.read_csv(
                    os.path.join(label_path, f'{str(frame).zfill(10)}.txt'),
                    header=None, delim_whitespace=True,
                    names=['x1', 'y1', 'x2', 'y2'])

            nao_bboxes = nao_bboxes.drop_duplicates()

            if is_show:
                img = cv2.imread(
                    os.path.join(img_path,
                                 f'{str(frame).zfill(10)}.jpg'))

            for index in sorted(anno_f.index):
                for box_ in anno_f.loc[index, 'bounding_boxes']:
                    x1 = int(box_[1])
                    y1 = int(box_[0])
                    x2 = int(box_[1] + box_[3])
                    y2 = int(box_[0] + box_[2])

                    for nao_index in sorted(nao_bboxes.index):
                        if ((x1 == nao_bboxes.loc[nao_index, 'x1']) &
                                (y1 == nao_bboxes.loc[nao_index, 'y1']) &
                                (x2 == nao_bboxes.loc[nao_index, 'x2']) &
                                (y2 == nao_bboxes.loc[nao_index, 'y2'])):
                            annos.loc[index, 'has_nao'] = 1
                            frames.append(frame)
                            ids.append(id)
                            labels.append(annos.loc[index, 'noun'])
                            nao_boxes.append([x1, y1, x2, y2])

                            if is_show:
                                cv2.rectangle(img, (x1, y1), (x2, y2),
                                              (0, 0, 255), 2)
                                object_label_ = annos.loc[index, 'noun']
                                cv2.putText(img, f'{object_label_}/{frame}',
                                            (x1, int(y1 + 30)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 2)

            if is_show:
                cv2.imshow(f'Frame', img)
                cv2.waitKey(1000)

        nao_dict['frame'] = frames
        nao_dict['id'] = ids
        nao_dict['label'] = labels
        nao_dict['nao_bbox'] = nao_boxes
        nao_annos = pd.DataFrame(nao_dict)
        nao_annos.to_csv(os.path.join(self.data_path, 'nao_annotations',
                                      f'nao_{str(id)}.csv'), index=False)
        # df = pd.read_csv(os.path.join(EpicData.data_path, 'nao_annotations',
        #                               f'nao_{str(id)}.csv'))

        annos.to_csv(os.path.join(self.data_path, 'nao_annotations',
                                  f'object_annos_{str(id)}.txt'),
                     columns=['noun_class', 'noun', 'participant_id', 'video_id',
                              'frame', 'bounding_boxes', 'id', 'has_nao'],
                     index=False)

    def label_merge(self):
        """if one video has been labeled more than one times."""
        pass


if __name__ == '__main__':
    EpicData = EpicDataParsing()
    # EpicData.data_annotation()

    for id in EpicData.labeled_ids:
        if not os.path.exists(os.path.join(EpicData.data_path,
                                           'nao_annotations',
                                           f'nao_{str(id)}.csv')):
            EpicData.label_transform(id, is_show=False)
            # EpicData.label_transform(id, is_show=True)

    # EpicData.label_transform('P04P04_13')
    # EpicData.label_transform('P04P04_02', is_show=False)
