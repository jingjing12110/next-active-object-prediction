# @File : test.py
# @Time : 2019/8/1 
# @Email : jingjingjiang2017@gmail.com 

import os
import torch.nn as nn
from torch.utils.data import DataLoader

from model.unet_resnet_hand_att import UNetResnetHandAtt
from metrics.metric import *
from data.adl import AdlDataset
from data.epic import EpicDataset
from opt import *


# exp_name = 'epic/unet_resnet'
exp_name = 'adl/unet_resnet_labeled2'
test_softmax = nn.Softmax(dim=-1)


def test(index, test_dataloader):
    model = UNetResnetHandAtt()

    model.load_state_dict(torch.load(os.path.join(args.exp_path, exp_name,
                                                  'ckpts',
                                                  f'model_epoch_{index}.pth'),
                                     map_location='cpu'))

    # if torch.cuda.is_available():
    #     model.load_state_dict(torch.load(os.path.join(args.exp_path, exp_name,
    #                                                   'ckpts',
    #                                                   f'model_epoch_{index}.pth')))
    # else:
    #     model.load_state_dict(torch.load(os.path.join(args.exp_path,
    #                                                   'model_test',
    #                                                   f'model_best_{index}.pth'),
    #                                      map_location='cpu'))
    model.cuda()

    model.eval()

    count_iter = 0
    p_all = []
    r_all = []
    f_all = []
    acc_all = []

    count_iter_th = 0
    p_all_th = []
    r_all_th = []
    f_all_th = []
    acc_all_th = []

    for i, data in enumerate(test_dataloader, start=1):
        img, mask = data
        outputs, feature_map = model(img.float().cuda(),
                                     with_output_feature_map=True)

        # 计算softmax
        out = outputs.view(outputs.shape[0], outputs.shape[1], -1).permute(0, 2, 1)
        out = test_softmax(out).permute(0, 2, 1)
        out = out.view(outputs.shape[0], outputs.shape[1], 224, -1)
        # use score threshold
        score = out.data.max(1)[0].cpu().numpy()
        x, y, z = np.where(score < 0.6)
        inds = outputs.data.max(1)[1].cpu().numpy()
        inds[x, y, z] = 0

        acc_, precision_, recall_, f1_score_ = compute_metrics(
            outputs.data.max(1)[1].cpu().numpy(), mask.float())
        acc_th, precision_th, recall_th, f1_score_th = compute_metrics(
            inds, mask.float())

        del outputs, mask

        if (not np.isnan(precision_)) & (not np.isnan(f1_score_)):
            acc_all.append(acc_)
            p_all.append(precision_)
            r_all.append(recall_)
            f_all.append(f1_score_)

            count_iter += 1

        if (not np.isnan(precision_th)) & (not np.isnan(f1_score_th)):
            acc_all_th.append(acc_th)
            p_all_th.append(precision_th)
            r_all_th.append(recall_th)
            f_all_th.append(f1_score_th)

            count_iter_th += 1

    ap = compute_ap(r_all, p_all)
    my_ap = my_compute_ap(r_all, p_all)
    acc = np.array(acc_all).mean()
    f_all = np.array(f_all).mean()
    precision = np.array(p_all).mean()
    recall = np.array(r_all).mean()

    print(f'index: {index}')
    print(f'ap: {ap:.4f}, f1_score: {f_all:.4f}, precision: {precision:.4f}, '
          f'recall: {recall:.4f}, acc: {acc:.4f}')
    print(f'mean ap: {my_ap}')
    print(f'frames of not being detecton: {i - count_iter}, '
          f'ratio: {1- (i -count_iter)/len(test_dataloader):.4f}')
    print('=====================================================================')

    my_ap_th = my_compute_ap(r_all_th, p_all_th)
    ap_th = compute_ap(r_all_th, p_all_th)
    acc_th = np.array(acc_all_th).mean()
    f_all_th = np.array(f_all_th).mean()
    precision_th = np.array(p_all_th).mean()
    recall_th = np.array(r_all_th).mean()

    print(f'index: {index}')
    print('threshold=0.6')
    print(
        f'ap: {ap_th:.4f}, f1_score: {f_all_th:.4f}, precision: {precision_th:.4f}, '
        f'recall: {recall_th:.4f}, acc: {acc_th:.4f}')
    print(f'mean ap: {my_ap_th}')
    print(f'frames of not being detecton: {i - count_iter_th}, '
          f'ratio: {1 - (i - count_iter_th) / len(test_dataloader):.4f}')
    print('=====================================================================')


def main():
    args.mode = 'test'
    test_data = AdlDataset(args)
    # test_data = EpicDataset(args)
    test_dataloader = DataLoader(test_data, batch_size=1,
                                 shuffle=True, num_workers=16,  # 4, 8, 16
                                 pin_memory=True)

    for index in {528, 530, 666}:
        test(index, test_dataloader)


def plot_PR(r_all, p_all):
    import pandas as pd
    import matplotlib.pyplot as plt
    rp = pd.DataFrame(np.array([r_all, p_all]).transpose(),
                      columns=['r_all', 'p_all'])
    rp = rp.sort_values(by='r_all').reset_index(drop=True)

    x = np.linspace(0.0, 1.0, 11)
    y = np.zeros(10)
    for i in range(10):
        tmp = rp[(rp['r_all'] > x[i]) & (rp['r_all'] < x[i+1])]
        if tmp.empty:
            y[i] = 0
        else:
            y[i] = tmp.p_all.max()

    plt.title(f'PR Curve: ResNet18UNet')
    plt.xlabel(f'recall')
    plt.ylabel(f'precision')
    plt.plot(rp.r_all.to_list(), rp.p_all.to_list())


if __name__ == '__main__':
    main()



