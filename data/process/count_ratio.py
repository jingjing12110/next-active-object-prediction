# @File : count_ratio.py
# @Time : 2019/7/26 
# @Email : jingjingjiang2017@gmail.com
import cv2

from data.adl import *


def isIntersection(xmin_a, xmax_a, ymin_a, ymax_a,
                   xmin_b, xmax_b, ymin_b, ymax_b):
    intersect_flag = True

    minx = max(xmin_a, xmin_b)
    miny = max(ymin_a, ymin_b)

    maxx = min(xmax_a, xmax_b)
    maxy = min(ymax_a, ymax_b)
    if minx > maxx or miny > maxy:
        intersect_flag = False
    return intersect_flag


def mat_inter(box1, box2):
    # 判断两个矩形是否相交
    # box=(xA,yA,xB,yB)
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2

    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)

    area1 = (x02 - x01) * (y02 - y01)
    area2 = (x12 - x11) * (y12 - y11)

    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
        col = min(x02, x12) - max(x01, x11)
        row = min(y02, y12) - max(y01, y11)
        intersection = col * row

        area = area1 + area2 - intersection

        return True, area
    else:
        area = area1 + area2

        return False, area


def count_adl_ratio():
    # from data.adl import *

    data_path = '/home/kaka/SSD_data/ADL'
    annos_path = 'ADL_annotations/object_annotation'
    action_path = 'ADL_annotations/action_annotation'
    frames_path = 'ADL_frames'
    key_frames_path = 'ADL_key_frames'

    video_id = {'P_01', 'P_02', 'P_03', 'P_04', 'P_05', 'P_06', 'P_07',
                'P_08', 'P_09', 'P_10', 'P_11', 'P_12', 'P_13', 'P_14',
                'P_15', 'P_16', 'P_17', 'P_18', 'P_19', 'P_20'}
    ratios = {}
    ratio_sum = 0.0

    for vi in sorted(video_id):
        print(f'start {vi}!')
        # img_path = os.path.join(data_path, frames_path, vi)
        # key_img_path = os.path.join(data_path, key_frames_path, vi)

        anno_name = 'object_annot_' + vi + '.txt'
        anno_path = os.path.join(data_path, annos_path, anno_name)
        annos = pd.read_csv(anno_path, header=None,
                            delim_whitespace=True, converters={0: str},
                            names=['object_track_id', 'x1', 'y1', 'x2', 'y2',
                                   'frame_id', 'is_active', 'object_label'])

        # print()
        # annos = annos[annos['is_active'] == 1]
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

        areas = 0.0
        count = 0
        count_3 = 0

        for i, frame_id in enumerate(annos.frame_id.unique(), start=1):
            df = annos[annos['frame_id'] == frame_id]
            if df.shape[0] == 1:
                area = (df.x2.iloc[0] - df.x1.iloc[0]
                        ) * (df.y2.iloc[0] - df.y1.iloc[0])

                areas += area
                count += 1

            if df.shape[0] == 2:
                box1 = (df.x1.iloc[0], df.y1.iloc[0],
                        df.x2.iloc[0], df.y2.iloc[0])
                box2 = (df.x1.iloc[1], df.y1.iloc[1],
                        df.x2.iloc[1], df.y2.iloc[1])

                _, area = mat_inter(box1, box2)

                areas += area
                count += 1

            if df.shape[0] >= 3:
                count_3 += 1

        ratio = areas / (307200 * count - areas)
        ratios[vi] = ratio
        ratio_sum = ratio_sum + ratio
        print(f'{vi}: {ratio:5f}')

    ratios_pd = pd.DataFrame.from_dict(ratios, orient='index',
                                       columns=['ratio'])
    ratios_pd = ratios_pd.reset_index()
    ratios_pd.to_csv('./adl_ratios.txt', sep=':', header=None, index=False)
    print("====================================================================")
    print(f'average is : {ratio_sum / 20.}')


def count_adl_ratio_v2():
    # adl_data = AdlDataset(args)  
    adl_data = AdlDatasetLabeled(args)
    adl_df = adl_data.data
    # adl_df = make_sequence_dataset(args)

    areas_ratio = 0.0
    areas = 0.
    count = 0
    count_3 = 0

    # for i, img_file in enumerate(adl_df.img_file.unique(), start=1):
    for i, img_file in enumerate(adl_df.img_file, start=1):
        df = adl_df[adl_df['img_file'] == img_file].reset_index(drop=True)
        # print(f'df.shape: {df.shape[0]}')
        if df.shape[0] == 1:
            bbox = df['bboxes'][0]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            areas_ratio += area / (307200 - area)
            areas += area
            count += 1

        elif df.shape[0] == 2:
            bbox1 = [df.bboxes[0][0], df.bboxes[0][1],
                     df.bboxes[0][2], df.bboxes[0][3]]
            bbox2 = [df.bboxes[1][0], df.bboxes[1][1],
                     df.bboxes[1][2], df.bboxes[1][3]]
            # bbox1 = df['bboxes'][0]
            # bbox2 = df['bboxes'][1]
            _, area = mat_inter(bbox1, bbox2)

            areas_ratio += area / (307200 - area)
            areas += area
            count += 1

        else:  # df.shape[0] >= 3:
            count_3 += 1

    print(f'i={i}, count={count}, count_3={count_3}')
    ratio = areas_ratio / count
    print(f'ratio: {ratio:.5f}')
    print(f'total: {areas / (307200 * count - areas)}')

    # ratios_pd = pd.DataFrame.from_dict(ratio, orient='index',
    #                                    columns=['ratio'])
    # ratios_pd = ratios_pd.reset_index()
    # ratios_pd.to_csv('./adl_ratios.txt', sep=':', header=None, index=False)
    # print("====================================================================")
    # print(f'average is : {ratio_sum / 20.}')


def count_epic_ratio_v2():
    from data.epic import make_sequence_dataset
    epic_df = make_sequence_dataset(args)
    areas_ratio = 0.0
    areas = 0.
    count = 0
    count_3 = 0

    # for i, img_file in enumerate(adl_df.img_file.unique(), start=1):
    for i, img_file in enumerate(epic_df.img_file, start=1):
        df = epic_df[epic_df['img_file'] == img_file].reset_index(drop=True)
        # print(f'df.shape: {df.shape[0]}')
        if df.shape[0] == 1:
            bbox = df['nao_bbox'][0]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

            areas_ratio += area / (2073600 - area)
            areas += area
            count += 1

        elif df.shape[0] == 2:
            bbox1 = [df.nao_bbox[0][0], df.nao_bbox[0][1],
                     df.nao_bbox[0][2], df.nao_bbox[0][3]]
            bbox2 = [df.nao_bbox[1][0], df.nao_bbox[1][1],
                     df.nao_bbox[1][2], df.nao_bbox[1][3]]
            _, area = mat_inter(bbox1, bbox2)

            areas_ratio += area / (2073600 - area)
            areas += area
            count += 1

        else:  # df.shape[0] >= 3:
            count_3 += 1

    print(f'i={i}, count={count}, count_3={count_3}')
    ratio = areas_ratio / count
    print(f'ratio: {ratio:.5f}')
    print(f'total: {areas / (2073600 * count - areas)}')


def count_epic_ratio():
    ratios = {}
    ratio_sum = 0.0

    for vi in sorted(id):
        print(f'start {vi}!')

        anno_name = 'nao_' + vi + '.csv'
        anno_path = os.path.join(args.data_path, annos_path, anno_name)
        annos = pd.read_csv(anno_path,
                            converters={"nao_bbox": literal_eval})

        if not annos.empty:
            img_path = os.path.join(args.data_path, frames_path,
                                    str(vi)[:3], str(vi)[3:])
            img_file = img_path + '/' + str(annos.loc[0, 'frame']).zfill(
                10) + '.jpg'
            img = cv2.imread(img_file)
            pixels = img.shape[0] * img.shape[1]

            areas = 0.0
            count = 0
            count_3 = 0

            for i, frame_id in enumerate(annos.frame.unique(), start=1):
                df = annos[annos['frame'] == frame_id]
                if df.shape[0] == 1:
                    area = (df.nao_bbox.iloc[0][2] - df.nao_bbox.iloc[0][0]
                            ) * (df.nao_bbox.iloc[0][3] - df.nao_bbox.iloc[0][1])

                    areas += area
                    count += 1

                if df.shape[0] == 2:
                    box1 = df.nao_bbox.iloc[0]
                    box2 = df.nao_bbox.iloc[1]

                    _, area = mat_inter(box1, box2)

                    areas += area
                    count += 1

                if df.shape[0] >= 3:
                    count_3 += 1

            ratio = areas / (pixels * count - areas)  # 正负样本比
            ratios[vi] = ratio
            ratio_sum = ratio_sum + ratio
            print(f'{vi}: {ratio:5f}')

    ratios_pd = pd.DataFrame.from_dict(ratios, orient='index',
                                       columns=['ratio'])
    ratios_pd = ratios_pd.reset_index()
    ratios_pd.to_csv('./epic_ratios.txt', sep=':', header=None, index=False)
    print("====================================================================")
    print(f'average is : {ratio_sum / 272.}')


if __name__ == '__main__':
    # count_epic_ratio()
    count_adl_ratio_v2()
    # count_epic_ratio_v2()
