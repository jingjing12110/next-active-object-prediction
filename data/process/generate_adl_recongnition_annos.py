# @File : adl_v0(未做sequence删除处理).py
# @Time : 2019/7/26 
# @Email : jingjingjiang2017@gmail.com
import os
import pandas as pd
import numpy as np

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

video_id = {'P_01', 'P_02', 'P_03', 'P_04', 'P_05', 'P_06', 'P_07',
            'P_08', 'P_09', 'P_10', 'P_11', 'P_12', 'P_13', 'P_14',
            'P_15', 'P_16', 'P_17', 'P_18', 'P_19', 'P_20'}


data_path = '/media/jing/YINGPAN/ADL'
annos_path = 'ADL_annotations/object_annotation'
action_path = 'ADL_annotations/action_annotation'
object_action_path = 'ADL_annotations/object_action_annotation'

# 将行为词编码为vector/ word2vec, 得到32 x 512 的vector

for vi in video_id:
    print(f"start {vi}!")
    action_anno_name = vi + '.txt'
    action_anno_path = os.path.join(data_path, action_path, action_anno_name)
    action_annos = pd.read_csv(action_anno_path, header=None,
                               delim_whitespace=True)
    action_annos = action_annos.iloc[:, 0:3]
    action_annos.columns = ['start_time', 'end_time', 'action_id']
    # action_annos = pd.read_csv(action_anno_path, header=None,
    #                            delim_whitespace=True,
    #                            # names=['start_time', 'end_time',
    #                            #        'action_id']
    #                            names=['start_time', 'end_time',
    #                                   'action_id', 'note']
    #                            )
    action_annos['start_frame'] = [30 * (60*int(a[0:2]) + int(a[3:5])) for a in
                                   action_annos.start_time]
    action_annos['end_frame'] = [30 * (60 * int(a[0:2]) + int(a[3:5])) for a in
                                 action_annos.end_time]
    
    anno_name = 'object_annot_' + vi + '.txt'
    anno_path = os.path.join(data_path, annos_path, anno_name)
    annos = pd.read_csv(anno_path, header=None,
                        delim_whitespace=True, converters={0: str},
                        names=['object_track_id', 'x1', 'y1', 'x2', 'y2',
                               'frame_id', 'is_active', 'object_label'])
    annos['action_id'] = np.repeat('0,', len(annos))
    
    for a in annos.frame_id:
        df = action_annos[(action_annos['start_frame'] <= a) & (
                    action_annos['end_frame'] >= a)]
        if not df.empty:
            b = annos[annos.frame_id == a]
            str_temp = ''
            for s in df.action_id.unique():
                str_temp += str(s) + ','
            annos.loc[annos.frame_id == a, 'action_id'] = np.repeat(str_temp,
                                                                    len(b))
            
        # print(df.shape)
    
    # annos_ = []
    # with open(anno_path, 'r') as f:
    #     for anno in f.readlines():
    #         anno = anno.split(" ", 4)
    #         # print(anno)
    #         annos_.append(anno)
    #
    # annos = pd.DataFrame(annos_, columns=['start_time', 'end_time',
    #                                       'action_id', 'note'])
    write_name = 'object_action_annot_' + vi + '.txt'
    write_path = os.path.join(data_path, object_action_path, write_name)
    annos.to_csv(write_path, sep=' ', header=None, index=False)
    
    print(f"finished {vi}")

