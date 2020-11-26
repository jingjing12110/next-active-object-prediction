# @File : annote_active_object_v2.py.py
# @Time : 2019/9/16
# @Email : jingjingjiang2017@gmail.com

import os

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

        # self.data_path = '/media/kaka/HD2T/dataset/EPIC_KITCHENS/data/'
        self.data_path = '/home/kaka/SSD_data/ADL/'
        self.img_path = os.path.join(self.data_path, 'ADL_key_frames')
        self.anno_path = os.path.join(self.data_path,
                                      'ADL_annotations/object_annotation')
        self.NAO_path = os.path.join(self.data_path,
                                     'ADL_annotations/active_annotations')

        self.box_scale = 30
        # self.labeled_ids = ['P_01', 'P_02', 'P_03', 'P_04', 'P_05', 'P_06',
        #                     'P_07', 'P_08', 'P_09', 'P_10', 'P_11', 'P_12',
        #                     'P_13', 'P_14', 'P_15', 'P_16', 'P_17', 'P_18',
        #                     'P_19', ]
        self.labeled_ids = []
        self.video_ids = ['P_01', 'P_02', 'P_03', 'P_04', 'P_05', 'P_06',
                          'P_07', 'P_08', 'P_09', 'P_10', 'P_11', 'P_12',
                          'P_13', 'P_14', 'P_15', 'P_16', 'P_17', 'P_18',
                          'P_19', 'P_20']

    def data_annotation(self):  # label all video
        for id in sorted(self.video_ids):
            if id not in self.labeled_ids:
                print(f'id: {id}')
                # video_path = os.path.join(self.NAO_path, id)
                # if not os.path.exists(video_path):
                #     os.mkdir(video_path)

                # df = self.annos[self.annos['id'] == id]
                anno_name = 'object_annot_' + id + '.txt'
                anno_path = os.path.join(self.anno_path, anno_name)
                df = pd.read_csv(anno_path, header=None,
                                 delim_whitespace=True, converters={0: str},
                                 names=['object_track_id', 'x1', 'y1', 'x2',
                                        'y2', 'frame_id', 'is_active',
                                        'object_label'])

                print(f'video {id} has {df.frame_id.unique().shape[0]} frames.')

                # self.data_annotation_video(df, id)

    def data_annotation_video(self, df, id):
        """标注单个视频"""
        img_src = os.path.join(self.img_path, id)
        frames = sorted(df.frame_id.unique())
        print(f'frame max is : {max(frames)}')
        # open bbox annotation
        cv2.namedWindow("Frame", 0)
        # for idx in range(len(frames)):
        idx = 0
        while idx < len(frames):
            print('=============================================')
            frame = frames[idx]
            image = cv2.imread(
                os.path.join(img_src, f'{str(frame + 1).zfill(6)}.jpg'))
            image = cv2.resize(image, (640, 480))
            # 求每帧bbox和画图
            ann_objects, image = self.df2objects(df, frame, image)

            cv2.imshow('Frame', image)
            flag_ = cv2.waitKey(0)

            if flag_ == 117:  # up
                print(f'key value in ASCII is: {flag_}/ up.')
                idx = idx - 1
                frame = frames[idx]
                image = cv2.imread(os.path.join(img_src,
                                                f'{str(frame + 1).zfill(6)}.jpg'))
                ann_objects, image = self.df2objects(df, frame, image)
                cv2.imshow('Frame', image)
                print(f'frame= {frame:06d}')
                # self.active_object_annotation(frame, image, ann_objects, pid)

            elif flag_ == 110:  # n
                print(f'key value in ASCII is: {flag_} /next.')
                idx = idx + 1
                frame = frames[idx]
                image = cv2.imread(os.path.join(img_src,
                                                f'{str(frame + 1).zfill(6)}.jpg'))
                ann_objects, image = self.df2objects(df, frame, image)
                cv2.imshow('Frame', image)
                print(f'frame= {frame:06d}')
                # self.active_object_annotation(frame, image, ann_objects, pid)

            elif flag_ == 99:  # c
                print(f'key value in ASCII is: {flag_} / current.')
                print(f'frame= {frame:06d}')
                print('labelling!!!!!!!!!!!!!!!!!!!!!!!!!')
                self.active_object_annotation(frame, image, ann_objects, id)

                idx = idx + 1

        cv2.destroyAllWindows()

    def df2objects(self, df, frame, image):
        df_f = df[df['frame_id'] == frame]
        colors = random_colors(50)
        ann_objects = []
        object_names = []

        for index in sorted(df_f.index):
            # for box in df_f.loc[index, 'bounding_boxes']:
            temp_ = [int(df_f.loc[index, 'x1']), int(df_f.loc[index, 'y1']),
                     int(df_f.loc[index, 'x2']), int(df_f.loc[index, 'y2'])]
            ann_objects.append(temp_)
            object_names.append(df_f.loc[index, 'object_label'])

        box_num = len(ann_objects)

        # 画bbox对应的点击框
        for j in range(box_num):
            box = ann_objects[j]
            b_lt = (box[0], box[1])
            b_rb = (box[2], box[3])
            color = colors[j]
            color255 = [i * 255 for i in color]
            cv2.rectangle(image, b_lt, b_rb, color255, 1)
            cv2.putText(image, f'{object_names[j]}/{frame}',
                        (box[0], int(box[1] + 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color255, 1)
            # this is novel that we draw a box with same color
            # with bbox to index it
            y1 = self.box_scale * j
            y2 = self.box_scale * (j + 1)
            x1 = 0
            x2 = 60
            # alpha = 0.5
            for c in range(3):
                image[y1:y2, x1:x2, c] = color255[c]
                cv2.putText(image, f'{object_names[j]}',
                            (x1+2, int(y1 + 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

        return ann_objects, image  # , df_f.iloc[0, 6] video id

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
                                   pid, f'{str(frame).zfill(6)}.txt'),
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
                                   pid, f'{str(frame).zfill(6)}.txt'),
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
                                   pid, f'{str(frame).zfill(6)}.txt'),
                      'w') as f_txt:
                f_txt.write(bbox_active_write)

    def label_transform(self, id, is_show=True):
        label_path = os.path.join(self.NAO_path, id)
        img_path = os.path.join(self.data_path,
                                'object_detection_images/bbox',
                                str(id)[:3], str(id)[3:])

        anno_name = 'object_annot_' + id + '.txt'
        anno_path = os.path.join(self.anno_path, anno_name)
        annos = pd.read_csv(anno_path, header=None,
                            delim_whitespace=True, converters={0: str},
                            names=['object_track_id', 'x1', 'y1', 'x2',
                                   'y2', 'frame_id', 'is_active',
                                   'object_label'])
        annos.insert(loc=8, column='is_next_active', value=0)

        nao_frames = os.listdir(label_path)
        nao_frames = [int(str(f).split('.')[0]) for f in nao_frames]

        print(f'{id} {annos.shape[0]} {len(nao_frames)}')

        for frame in sorted(nao_frames):
            anno_f = annos[annos['frame_id'] == frame]
            nao_bboxes = pd.read_csv(os.path.join(label_path,
                                                  f'{str(frame).zfill(6)}.txt'),
                                     header=None, delim_whitespace=True,
                                     names=['x1', 'y1', 'x2', 'y2'])
            nao_bboxes = nao_bboxes.drop_duplicates()

            if is_show:
                img = cv2.imread(
                    os.path.join(img_path,
                                 f'{str(frame).zfill(10)}.jpg'))

            for index in sorted(anno_f.index):  # 数据集给的labeled object
                x1 = int(anno_f.loc[index, 'x1'])
                y1 = int(anno_f.loc[index, 'y1'])
                x2 = int(anno_f.loc[index, 'x2'])
                y2 = int(anno_f.loc[index, 'y2'])

                for nao_index in sorted(nao_bboxes.index):
                    if ((x1 == nao_bboxes.loc[nao_index, 'x1']) & (
                            y1 == nao_bboxes.loc[nao_index, 'y1']) & (
                            x2 == nao_bboxes.loc[nao_index, 'x2']) & (
                            y2 == nao_bboxes.loc[nao_index, 'y2'])):
                        annos.loc[index, 'is_next_active'] = 1

        annos.to_csv(os.path.join(self.data_path,
                                  'ADL_annotations/nao_annotation',
                                  f'nao_{str(id)}.txt'),
                     sep=' ', header=None, index=False)

    def label_merge(self):
        """if one video has been labeled more than one times."""
        pass


if __name__ == '__main__':
    EpicData = EpicDataParsing()
    EpicData.data_annotation()
    # EpicData.label_transform('P_20', is_show=False)
