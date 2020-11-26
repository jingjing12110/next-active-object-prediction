# @File : cam.py.py 
# @Time : 2019/10/31 
# @Email : jingjingjiang2017@gmail.com 
import os

import cv2
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from data.adl import AdlDatasetV2
from data.epic import EpicDatasetV2
from metrics.metric import *
from model.unet_resnet_hand_att import UNetResnetHandAtt
from opt import *

exp_name = 'epic/unet_resnet_hand_att'
# exp_name = 'adl/unet_resnet_hand_att_labeled'
test_softmax = nn.Softmax(dim=-1)


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

hm_transform = transforms.Compose([
    transforms.ToTensor()])


# save_path = '/home/kaka/SSD_data/ADL/results'
save_path = '/media/kaka/HD2T/dataset/EPIC_KITCHENS/results'


def cam(epoch):
    model = UNetResnetHandAtt()

    model.cuda(device=args.device_ids[0])
    # model = torch.nn.DataParallel(model, device_ids=args.device_ids)

    # model.load_state_dict(torch.load(os.path.join(args.exp_path, exp_name,
    #                                               'ckpts',
    #                                               f'model_epoch_{index}.pth'),
    #                                  map_location='cpu'))

    checkpoint = torch.load(os.path.join(args.exp_path, exp_name, 'ckpts/',
                                         f'model_epoch_{epoch}.pth'),
                            map_location='cpu')
    model.load_state_dict(checkpoint['net'])

    # model = model.cpu()
    model = model.cuda()

    model.eval()

    # adl_data = AdlDatasetV2(args)
    # adl_df = adl_data.data
    # hand_hms = adl_data.hand_hms
    epic_data = EpicDatasetV2(args)
    adl_df = epic_data.data
    hand_hms = epic_data.hand_hms

    for index in adl_df.index:
        if index % 50 == 0:
            print(f'index: {index}')
            
        img_file = adl_df.loc[index, 'img_file']
        video_id = img_file[-21:-15]
        # video_id = img_file[-15:-11]
        img_src = os.path.join(save_path, f'{video_id}')

        if not os.path.exists(img_src):
            os.mkdir(img_src)

        if not os.path.exists(os.path.join(img_src, f'{img_file[-14:]}')):
            img = Image.open(img_file).convert('RGB')
            img = img.resize((320, 224), Image.ANTIALIAS)
            img = img_transform(img)

            h_hm = np.array(hand_hms[img_file])
            h_hm = hm_transform(h_hm)

            outputs = model(img.unsqueeze(dim=0).cuda(), h_hm.unsqueeze(dim=0).cuda())

            # compute softmax
            out = outputs.view(outputs.shape[0],
                               outputs.shape[1], -1).permute(0, 2, 1).contiguous()
            out = test_softmax(out).permute(0, 2, 1)
            out = out.view(outputs.shape[0], outputs.shape[1], 224, -1)

            out1 = out.squeeze(dim=0).detach().cpu().numpy()[1, :, :]  # nao label=1

            x, y = np.where(out1[:, :] < 0.6)
            out1[x, y] = 0

            heatmap = cv2.applyColorMap(np.uint8(255 * out1), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255

            im = cv2.imread(img_file)
            im = cv2.resize(im, (320, 224)) / 255

            cam = heatmap * 0.3 + im * 0.7
            cam = (cam - cam.min())/(cam.max() - cam.min())

            cv2.imwrite(os.path.join(img_src, f'{img_file[-14:]}'), cam * 255)
            # cv2.imwrite(os.path.join(img_src, f'{img_file[-10:]}'), cam * 255)
            # plt.imshow(cam, cmap='jet')
            # plt.xticks([])
            # plt.yticks([])
            # plt.savefig(os.path.join(img_src, f'{img_file[-14:]}'),
            #             bbox_inches='tight', dpi=120, pad_inches=0.0)


def main():
    for index in {551}:    # EPIC
    # for index in {710}:      # ADL
        cam(index)


if __name__ == '__main__':
    main()



