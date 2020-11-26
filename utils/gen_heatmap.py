# @File : gen_heatmap.py 
# @Time : 2019/10/17 
# @Email : jingjingjiang2017@gmail.com 
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from model.unet_resnet import UNetResNet18
from opt import args

ckpt_path = '/media/kaka/HD2T/code/next_active_object/experiments/epic/unet_resnet/ckpts'

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])


class ClassSaliencyMap(object):
    def __init__(self):
        super(ClassSaliencyMap, self).__init__()

        self.model = UNetResNet18()
        self.model.load_state_dict(torch.load(
            os.path.join(ckpt_path, f'model_epoch_174.pth')))

        # self.backprop = Backprop(self.model)

        self.data_path = os.path.join(args.data_path, f'epic_train_df.csv')
        # self.img_path = os.path.join(args.data_path, frames_path)
        self.data = pd.read_csv(self.data_path)

    def compute_saliency_map(self, X, y):
        self.model.eval()

        X_var = Variable(X, requires_grad=True)
        y_var = Variable(y)

        scores = self.model(X_var)

        # 得到正确类的分数，scores为[5]的Tensor
        scores = scores.gather(1, y_var.view(-1, 1)).squeeze()

        # 反向计算，从输出的分数到输入的图像进行一系列梯度计算
        scores.backward(torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0]))

        # 得到正确分数对应输入图像像素点的梯度
        saliency = X_var.grad.data

        saliency = saliency.abs()  # 取绝对值
        saliency, i = torch.max(saliency, dim=1)
        saliency = saliency.squeeze()  # 去除1维

        return saliency

    def show_heat_map(self):
        for index in self.data.index:
            img_path = self.data.loc[index, 'img_file']
            img = Image.open(img_path).convert('RGB')
            img = img.resize((320, 224), Image.ANTIALIAS)
            img = TRANSFORM(img).unsqueeze(dim=0)

            out, fea = self.model(img, with_output_feature_map=True)
            f1 = fea[0, 0:3, :, :]
            # img = load_image(img_path)
            # input_ = apply_transforms(img)
            # # target_class = 1
            #
            # self.backprop.visualize(input_, guided=True)
            frame = cv2.imread(img_path)
            labels = self.data.loc[index, 'hand_bbox']
            accumulated_exposures = np.zeros((frame.shape[0], frame.shape[1]),
                                             dtype=np.float)

            seconds_per_frame = 1.0
            maskimg = np.zeros(accumulated_exposures.shape, dtype=np.float)
            for label in labels:
                cv2.fillConvexPoly(maskimg, label, (seconds_per_frame))

            highlighted_image = highlight_labels(frame, labels, maskimg)
            cv2.imwrite('output/%s' % os.path.basename(frame_path),
                        highlighted_image)

            # accumulate the heatmap object exposure time
            accumulated_exposures = accumulated_exposures + maskimg

    def plot_heat_map(self, fea, out):
        fea = fea[0, 0:3, :, :]
        fea = fea.detach().cpu().numpy().transpose(1, 2, 0) / fea.max()
        plt.imshow(fea)
        plt.show()

        o1 = out.data.max(1)[0].cpu().numpy()[0, :, :]
        o1 = (o1 - o1.min()) / (o1.max() - o1.min())
        o1[o1 < 0.2] = 0
        plt.imshow()
        plt.show()


if __name__ == '__main__':
    SaliencyMap = ClassSaliencyMap()

    SaliencyMap.show_heat_map()
