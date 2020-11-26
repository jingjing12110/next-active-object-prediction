# @File : train.py
# @Time : 2019/10/18 
# @Email : jingjingjiang2017@gmail.com 

import os
from datetime import datetime

from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader

from data.adl import AdlDatasetV2
from data.epic import EpicDatasetV2
from metrics.losses import *
from metrics.metric import *
from model.unet_resnet_hand_att import UNetResnetHandAtt
from opt import *

###########################################################################
# os.environ['CUDA_VISIBLE_DEVICES'] = '4, 5'
str_ids = args.gpu_ids.split(',')
gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)
# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
# device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
###########################################################################


SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

current_epoch = 436
exp_name = args.exp_name

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
writer_train = SummaryWriter(os.path.join(args.exp_path, exp_name, 'logs',
                                          f'train/{TIMESTAMP}'))
writer_val = SummaryWriter(os.path.join(args.exp_path, exp_name, 'logs',
                                        f'val/{TIMESTAMP}'))

multi_gpu = False


def main():
    model = UNetResnetHandAtt()
    # load parameters                      
    model.load_state_dict(torch.load(
        os.path.join(args.exp_path, exp_name,
                     f'ckpts/model_epoch_{current_epoch}.pth')))
    
    # for p in model.base_model.parameters():
    #     p.requires_grad = False
    
    model.cuda(device=args.device_ids[0])
    
    if args.dataset == 'EPIC':
        train_data = EpicDatasetV2(args)
        train_dataloader = DataLoader(train_data, batch_size=args.bs,
                                      shuffle=True, num_workers=16,
                                      pin_memory=True)
        
        val_args = args
        val_args.mode = 'val'
        val_data = EpicDatasetV2(val_args)
        val_dataloader = DataLoader(val_data,
                                    batch_size=val_args.bs,
                                    shuffle=True, num_workers=16,
                                    pin_memory=True)
    else:
        train_data = AdlDatasetV2(args)
        train_dataloader = DataLoader(train_data, batch_size=args.bs,
                                      shuffle=True, num_workers=16,
                                      pin_memory=True)
        
        val_args = args
        val_args.mode = 'val'
        val_data = AdlDatasetV2(val_args)
        val_dataloader = DataLoader(val_data,
                                    batch_size=val_args.bs,
                                    shuffle=True, num_workers=16,
                                    pin_memory=True)
    
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           betas=(0.9, 0.99),
                           weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.8,
                                                     patience=3,
                                                     verbose=True,
                                                     min_lr=0.0000001)
    
    if multi_gpu:
        optimizer = nn.DataParallel(optimizer, device_ids=args.device_ids)
    # else:
    # UNet的criterion
    if args.dataset == 'EPIC':
        class_weights = torch.FloatTensor([1, 11.2]).cuda(args.device_ids[0])
    else:
        class_weights = torch.FloatTensor([1, 9.35]).cuda(args.device_ids[0])
    criterion = nn.CrossEntropyLoss(class_weights)
    # criterion = FocalLoss()
    
    train_args['ckpt_path'] = os.path.join(train_args['exp_path'],
                                           exp_name, 'ckpts/')
    if not os.path.exists(train_args['ckpt_path']):
        os.mkdir(train_args['ckpt_path'])
    
    write_val = open(os.path.join(train_args['ckpt_path'], 'val.txt'), 'w')
    
    for epoch in range(current_epoch + 1, train_args['epochs'] + 1):
        print('=================================================================')
        val_loss = val(val_dataloader, model, criterion, epoch - 1, write_val)
        # scheduler.step(val_loss)
        print('=================================================================')
        
        train(train_dataloader, model, criterion, optimizer, epoch, train_args)
    
    writer_train.close()
    write_val.close()


def train(train_dataloader, model, criterion, optimizer, epoch, train_args):
    train_losses = 0.
    curr_iter = (epoch - 1) * len(train_dataloader)
    
    for i, data in enumerate(train_dataloader, start=1):
        img, mask, hand_hm = data
        img = Variable(img.float().cuda(args.device_ids[0]))
        hand_hm = Variable(hand_hm.float().cuda(args.device_ids[0]))
        # forward
        outputs = model(img, hand_hm)
        # outputs = model(hand_hm)
        del img, hand_hm
        
        loss = criterion(outputs, mask.long().cuda(args.device_ids[0]))
        del outputs, mask
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        # optimizer.step()
        if multi_gpu:
            optimizer.module.step()
        else:
            optimizer.step()
        train_losses += loss.item()
        
        curr_iter += 1
        writer_train.add_scalar("train_loss", train_losses / i, curr_iter)
        
        if i % train_args['print_every'] == 0:
            print(f"[epoch {epoch}], [iter {i} / {len(train_dataloader)}], "
                  f"[train loss {train_losses / i:5f}]")
    
    torch.save(model.state_dict(),
               os.path.join(train_args['ckpt_path'],
                            f'model_epoch_{epoch}.pth'))


def val(val_dataloader, model, criterion, epoch, write_val):
    model.eval()
    val_loss = AverageMeter()
    targets_all, predictions_all = [], []
    
    for data in val_dataloader:
        img, mask, hand_hm = data
        n = img.size(0)
        img = Variable(img.float().cuda(args.device_ids[0]))
        hand_hm = Variable(hand_hm.float().cuda(args.device_ids[0]))
        mask = mask.long().cuda(args.device_ids[0])
        # forward
        outputs = model(img, hand_hm)
        # outputs = model(hand_hm)
        del img, hand_hm
        
        predictions_all.append(outputs.data.max(1)[1].cpu().numpy())
        targets_all.append(mask.data.cpu().squeeze_(0))
        
        # loss = criterion(outputs.permute(0, 2, 3, 1).reshape([-1, 2]),
        #                  mask.flatten())
        loss = criterion(outputs, mask)
        val_loss.update(loss.item(), n)
        
        del outputs, mask
    
    acc_, precision, recall, f1_score_ = compute_metrics(predictions_all,
                                                         targets_all)
    del targets_all, predictions_all
    print(f'[epoch {epoch}], [val loss {val_loss.avg:5f}], [acc {acc_:5f}], '
          f'[precision {precision:5f}], [recall {recall:5f}], '
          f'[f1_score {f1_score_:5f}]')
    
    write_val.writelines(f"[epoch {epoch}], "
                         f"[acc {acc_:5f}], [precision {precision:5f}], "
                         f"[recall {recall:5f}], [f1_score {f1_score_:5f}]]\n")
    
    writer_val.add_scalar('val_loss', val_loss.avg, epoch)
    writer_val.add_scalar('acc', acc_, epoch)
    writer_val.add_scalar('precision', precision, epoch)
    writer_val.add_scalar('recall', recall, epoch)
    writer_val.add_scalar('f1_score', f1_score_, epoch)
    
    model.train()
    
    return val_loss.avg


if __name__ == '__main__':
    main()
