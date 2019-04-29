import numpy as np
import pickle
import os
from PIL import Image
import time
from tqdm import tqdm
import shutil
from random import randint
import argparse
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils import *
from network import *

parser = argparse.ArgumentParser(description='UCF101 spatial stream on resnet101')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=32, type=int, metavar='N',
                    help='mini-batch size (default: 25)')
parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--linux', default=False, type=bool, metavar='N', help='set OS')
parser.add_argument('--use-gpus', default='0', type=str, metavar='GPU', help='set GPUs used')


class SpatialModel(object):
    def __init__(self):
        self.model = Spatial_CNN(
            resume='src/two-stream-action-recognition/record/spatial/checkpoint.pth.tar')

    def run(self, path, img_num):
        validation_set = spatial_dataset(img_num=img_num, root_dir=path,
                                         transform=transforms.Compose([
                                             transforms.Scale([224, 224]),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])
                                         ]))

        self.model.test_loader = DataLoader(
            dataset=validation_set,
            batch_size=16,
            shuffle=False,
            num_workers=2)
        return self.model.run()


class Spatial_CNN():
    def __init__(self, resume):
        self.resume = resume
        self.test_loader = None
        self.epoch = 0

    def build_model(self):
        print('==> Build model and setup loss and optimizer')
        # build model
        self.model = resnet101(pretrained=True, channel=3).cuda()

    def resume_and_evaluate(self):
        if os.path.isfile(self.resume):
            print("==> loading checkpoint '{}'".format(self.resume))
            checkpoint = torch.load(self.resume)
            self.model.load_state_dict(checkpoint['state_dict'])
            print("==> loaded checkpoint '{}'".format(self.resume))
        else:
            print("==> no checkpoint found at '{}'".format(self.resume))
            return None

        return self.validate_1epoch()

    def run(self):
        self.build_model()
        return self.resume_and_evaluate()

    def validate_1epoch(self):
        batch_time = AverageMeter()
        # switch to evaluate mode
        self.model.eval()
        self.dic_video_level_preds = {}
        end = time.time()
        predicts = np.zeros([101], np.float32)

        progress = tqdm(self.test_loader)

        for i, data in enumerate(progress):
            data_var = Variable(data, volatile=True).cuda(async=True)

            # compute output
            output = self.model(data_var)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # Calculate video level prediction
            preds = output.data.cpu().numpy()
            del output
            predicts += preds.sum(axis=0)
        top5 = predicts.argsort()[-5:][::-1]
        info = {'Epoch': [self.epoch],
                'Batch Time': [round(batch_time.avg, 3)],
                'Prec@1': top5[0],
                'Prec@5': top5}
        print(info)
        # record_info(info, 'record/spatial/rgb_test.csv', 'test')
        return top5


class spatial_dataset(Dataset):
    def __init__(self, img_num, root_dir, transform=None):
        self.img_num = img_num
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return self.img_num

    def load_ucf_image(self, index):
        path = self.root_dir
        img = Image.open(open(path + 'frame' + str(index + 1).zfill(6) + '.jpg', 'rb'))
        transformed_img = self.transform(img)
        img.close()

        return transformed_img

    def __getitem__(self, idx):
        data = self.load_ucf_image(idx)
        return data
