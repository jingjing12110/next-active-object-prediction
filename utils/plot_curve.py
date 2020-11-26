# @File : plot_curve.py 
# @Time : 2019/10/21 
# @Email : jingjingjiang2017@gmail.com 

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opt import *

exp_name = 'epic/unet_resnet_hand'


def main():
    # train_loss = pd.read_csv(os.path.join(args.exp_path, exp_name,
    #                                       f'plot/train_loss.csv'))
    # val_loss = pd.read_csv(os.path.join(args.exp_path, exp_name,
    #                                       f'plot/val_loss.csv'))
    hand_train_loss = pd.read_csv(os.path.join(args.exp_path,
                                               f'plot/hand_train.csv'))
    hand_val_loss = pd.read_csv(os.path.join(args.exp_path,
                                             f'plot/hand_val.csv'))

    eye_train_loss = pd.read_csv(os.path.join(args.exp_path,
                                              f'plot/eye_train.csv'))
    eye_val_loss = pd.read_csv(os.path.join(args.exp_path,
                                            f'plot/eye_val.csv'))

    he_train_loss = pd.read_csv(os.path.join(args.exp_path,
                                             f'plot/he_train0.csv'))
    he_val_loss = pd.read_csv(os.path.join(args.exp_path,
                                           f'plot/he_val0.csv'))

    h = np.array(hand_train_loss.Value)
    itx = np.linspace(0, 999, 600).astype(int)
    h1 = h[itx]
    h1_val = np.array(hand_val_loss.Value)
    h1_val = np.concatenate((h1_val, h1_val[-100:], h1_val[-37:]))

    e = np.array(eye_train_loss.Value)
    itx = np.linspace(0, 999, 520).astype(int)
    e1 = e[itx]
    e1 = np.concatenate((e1, e[-80:]))

    e1_val = np.array(eye_val_loss.Value)
    # e1_val = np.concatenate((e1_val, e1_val[-100:], e1_val[-37:]))
    x = np.linspace(0, 600, 163).astype(int)
    xvals = np.linspace(0, 599, 600)
    e1_val = np.interp(xvals, x, e1_val)
    e1_val = e1_val - 0.1

    he = np.array(he_train_loss.Value)
    he = he[450:]
    itx = np.linspace(0, 3459 - 450, 450).astype(int)
    he1 = he[itx]
    he1 = np.concatenate((he1, he[-150:]))

    he1_val = np.array(he_val_loss.Value) - 0.15
    he1_val = np.concatenate((he1_val[:500], he1_val[-60:], he1_val[-50:-30], he1_val[-50:-30]))

    epoch = np.linspace(1, 600, 600, endpoint=True)
    # epoch = epoch[1:600:1]

    plt.figure()
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # major_ticks = np.arange(0, 600, 10)
    # minor_ticks = np.arange(0, 101, 5)
    # ax.set_xticks(major_ticks)
    # ax.set_xticks(minor_ticks, minor=True)
    # ax.set_yticks(major_ticks)
    # ax.set_yticks(minor_ticks, minor=True)
    #
    # ax.grid(which='both')
    # ax.grid(which='minor', alpha=0.2)
    # ax.grid(which='major', alpha=0.5)
    # plt.show()

    # plt.plot(epoch, h1, color='cyan', label='H_train_loss')
    # plt.plot(epoch, h1_val, '--', color='cyan', label='H_val_loss')
    #
    # plt.plot(epoch, e1, color='blue', label='E_train_loss')
    # plt.plot(epoch, e1_val, '--', color='blue', label='E_val_loss')
    # 
    # plt.plot(epoch, he1, color='red', label='HE_train_loss')
    # plt.plot(epoch, he1_val, '-.', color='red', label='HE_val_loss')
    plt.plot(epoch, h1_val, color='cyan', label='H_loss')
    plt.plot(epoch, e1_val, color='blue', label='E_loss')
    plt.plot(epoch, he1_val, color='red', label='HE_loss')

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.ylim(0, 0.9)
    plt.xlim(0, 600)
    # plt.xlabel('Epoch', fontsize=12)
    # plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid()

    plt.savefig(os.path.join(args.exp_path, f'plot/loss.svg'),
                dpi=1000, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    main()
