from spatial_cnn import Spatial_CNN
from motion_cnn import Motion_CNN

import numpy as np
import pickle
import os
from PIL import Image
import time
from tqdm import tqdm
import shutil
from random import randint
import argparse

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

import dataloader
from utils import *
from network import *

parser = argparse.ArgumentParser(description='UCF101 spatial stream on resnet101')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=8, type=int, metavar='N',
                    help='mini-batch size (default: 25)')
parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--spatial_resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--temporal_resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--linux', default=False, type=bool, metavar='N', help='set OS')
parser.add_argument('--use-gpus', default='0', type=str, metavar='GPU', help='set GPUs used')


def main():
    global arg
    arg = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = arg.use_gpus

    spatial_loader = dataloader.spatial_dataloader(
        BATCH_SIZE=arg.batch_size,
        num_workers=4,
        path='E:\\dataset\\ucf101\\jpegs_256\\',
        ucf_list='E:\\Graduate\\formers\\two-stream-action-recognition\\UCF_list\\',
        ucf_split='01',
    )
    temporal_loader = dataloader.Motion_DataLoader(
        BATCH_SIZE=arg.batch_size,
        num_workers=4,
        path='E:\\dataset\\ucf101\\tvl1_flow\\',
        ucf_list='E:\\Graduate\\formers\\two-stream-action-recognition\\UCF_list\\',
        ucf_split='01',
        in_channel=10,
    )

    temporal_train, temporal_test, _ = temporal_loader.run()
    spatial_train, spatial_test, test_video = spatial_loader.run()

    spatial_model = Spatial_CNN(
        nb_epochs=arg.epochs,
        lr=arg.lr,
        batch_size=arg.batch_size,
        resume=arg.spatial_resume,
        start_epoch=arg.start_epoch,
        evaluate=arg.evaluate,
        train_loader=spatial_train,
        test_loader=spatial_test,
        test_video=test_video
    )

    temporal_model = Motion_CNN(
        # Data Loader
        train_loader=temporal_train,
        test_loader=temporal_test,
        # Utility
        start_epoch=arg.start_epoch,
        resume=arg.temporal_resume,
        evaluate=arg.evaluate,
        # Hyper-parameter
        nb_epochs=arg.epochs,
        lr=arg.lr,
        batch_size=arg.batch_size,
        channel=10 * 2,
        test_video=test_video
    )

    spatial_model.build_model()
    spatial_model.resume_model()

    temporal_model.build_model()
    temporal_model.resume_model()

    print('==> Epoch:[{0}/{1}][validation stage]'.format(0, arg.epochs))
    batch_time = AverageMeter()
    # losses = AverageMeter()
    # top1 = AverageMeter()
    # top5 = AverageMeter()
    # switch to evaluate mode
    spatial_model.model.eval()
    temporal_model.model.eval()
    dic_video_level_preds = {}
    end = time.time()
    print(len(spatial_test), len(temporal_test))
    spatial_it = iter(spatial_test)
    temporal_it = iter(temporal_test)

    for i in range(len(spatial_test)):
    # for i in range(100):
        print("{}%, {}/{}, [{}/it]".format(i / len(spatial_test) * 100, i, len(spatial_test),
                                           batch_time.avg))
        (keys, data, label) = spatial_it.next()
        (k, d, l) = temporal_it.next()
        data_var = Variable(data, volatile=True).cuda(async=True)
        d_var = Variable(d, volatile=True).cuda(async=True)

        # compute output
        output = spatial_model.model(data_var)
        o = temporal_model.model(d_var)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # Calculate video level prediction
        preds = output.data.cpu().numpy()
        p = o.data.cpu().numpy()
        del output
        del o
        preds += p

        nb_data = preds.shape[0]
        for j in range(nb_data):
            videoName = keys[j].split('/', 1)[0]
            # videoName = keys[j]
            if videoName not in dic_video_level_preds.keys():
                dic_video_level_preds[videoName] = preds[j, :]
            else:
                dic_video_level_preds[videoName] += preds[j, :]
        del keys, data, label, data_var
        del k, d, l, d_var
    video_top1, video_top5, video_loss = spatial_model.frame_to_video_level_accuracy(
        dic_video_level_preds)

    info = {'Epoch': [0],
            'Batch Time': [round(batch_time.avg, 3)],
            'Loss': [round(video_loss, 5)],
            'Prec@1': [round(video_top1, 3)],
            'Prec@5': [round(video_top5, 3)]}
    record_info(info, 'record/spatial/rgb_test.csv', 'test')
    # return video_top1, video_loss


if __name__ == '__main__':
    main()
