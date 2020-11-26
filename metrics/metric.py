import numpy as np
import torch


# confusion matrix
def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    label = num_classes * label_true[mask] + label_pred[mask]
    hist = np.bincount(label, minlength=num_classes ** 2).reshape(num_classes,
                                                                  num_classes)
    return hist


def compute_metrics_v2(predictions, targets, num_classes=2):
    """
    compute metrics of active object
    :param predictions:
    :param targets:
    :param num_classes:
    :return:
    """
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, targets):
        hist += _fast_hist(lp, lt, 2)

    acc = np.diag(hist).sum() / hist.sum()  # pixel accurency
    precision = (np.diag(hist) / hist.sum(axis=0))[0]  # active
    # acc_cls = np.nanmean(acc_cls)  # 忽略acc_cls的nan

    # Recall
    recall = (np.diag(hist) / hist.sum(axis=1))[0]

    f1_score = 2 * precision * recall / (precision + recall)

    # iu_ = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    # # mean_iu = np.nanmean(iu)
    # iu = iu_[1]
    # freq = hist.sum(axis=1) / hist.sum()
    # fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return acc, precision, recall, f1_score


# 语义分割的评价指标
def evaluate(predictions, targets, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, targets):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # axis 0: gt, axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()  # 所有类别分类正确的acc
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)  # 忽略acc_cls的nan
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return acc, acc_cls, mean_iu, fwavacc


class Evaluator(object):
    def __init__(self, gt_image, pre_image, num_class=2):
        self.gt_image = gt_image
        self.pre_image = pre_image
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def pixel_accuracy(self):
        acc_pa = np.diag(
            self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return acc_pa

    def pixel_accuracy_class(self):
        acc_mpa = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(
            axis=1)
        acc_mpa = np.nanmean(acc_mpa)
        return acc_mpa

    def mean_intersection_over_union(self):
        mIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) +
                np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        mIoU = np.nanmean(mIoU)
        return mIoU

    def frequency_weighted_intersection_over_union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(
            self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) +
                np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        fwIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return fwIoU

    def _generate_matrix(self):
        mask = (self.gt_image >= 0) & (self.gt_image < self.num_class)
        label = self.num_class * self.gt_image[mask].astype('int') \
                + self.pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self):
        assert self.gt_image.shape == self.pre_image.shape
        self.confusion_matrix += self._generate_matrix()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def evaluate(self):
        return self.pixel_accuracy(), self.pixel_accuracy_class(), \
               self.mean_intersection_over_union(), \
               self.frequency_weighted_intersection_over_union()


# AP
def compute_ap(recall, precision, use_11_points=False):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    if use_11_points:
        recall = np.array(recall)
        precision = np.array(precision)
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                # 取最大值，保证precise非减
                p = np.max(precision[recall >= t])
            ap = ap + p / 11.

    else:
        # correct AP calculation
        # first append sentinel values at the end
        # 取所有不同的recall对应的点处的精度值做平均
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))

        # compute the precision envelope
        # 计算包络线，从后往前取最大保证precise非减
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        # 找出所有检测结果中recall不同的点
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        # 用recall的间隔对精度作加权平均
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def confusion_matrix(pred, target, num_classes=2):
    con_mat = np.zeros((num_classes, num_classes))
    con_mat[0][0] = (pred * target).sum()
    con_mat[0][1] = target.sum() - con_mat[0][0]
    con_mat[1][0] = pred.sum() - con_mat[0][0]
    con_mat[1][1] = ((pred - 1) * (target - 1)).sum()

    return con_mat


def compute_metrics(predictions, targets, num_classes=2):
    """
    compute metrics of active object
    :param predictions:
    :param targets:
    :param num_classes:
    :return:
    """
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, targets):
        hist += confusion_matrix(lp, lt.numpy())

    acc = np.diag(hist).sum() / hist.sum()  # pixel accurency
    precision = (np.diag(hist) / hist.sum(axis=0))[0]  # active
    # acc_cls = np.nanmean(acc_cls)  # 忽略acc_cls的nan

    # Recall
    recall = (np.diag(hist) / hist.sum(axis=1))[0]

    f1_score = 2 * precision * recall / (precision + recall)

    # iu_ = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    # # mean_iu = np.nanmean(iu)
    # iu = iu_[1]
    # freq = hist.sum(axis=1) / hist.sum()
    # fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    return acc, precision, recall, f1_score


def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:  # 使用07年方法
        # 11 个点
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])  # 插值
            ap = ap + p / 11.
    else:  # 新方式，计算所有点
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision 曲线值（也用了插值）
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_ap2(rec, prec):
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]

    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def my_compute_ap(r_all, p_all):
    import pandas as pd
    rp = pd.DataFrame([r_all, p_all]).transpose()
    rp.columns = ['r', 'p']

    ap_tmp = []
    r_step = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for i in range(10):
        tmp = rp[(rp['r'] > r_step[i]) & (rp['r'] <= r_step[i + 1])]
        ap_tmp.append(tmp['p'].mean())

    return np.nanmean(np.array(ap_tmp))
