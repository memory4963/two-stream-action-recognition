import pickle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
from dataloader.split_train_test_video import *
from skimage import io, color, exposure


class spatial_dataset(Dataset):
    def __init__(self, img_num, root_dir, transform=None):
        self.img_num = img_num
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return self.img_num

    def load_ucf_image(self, index):
        path = self.root_dir
        img = Image.open(open(path + 'frame' + str(index).zfill(6) + '.jpg', 'rb'))
        transformed_img = self.transform(img)
        img.close()

        return transformed_img

    def __getitem__(self, idx):
        data = self.load_ucf_image(idx)
        return data


class spatial_dataloader():
    def __init__(self, BATCH_SIZE, num_workers, path):
        self.BATCH_SIZE = BATCH_SIZE
        self.num_workers = num_workers
        self.data_path = path
        # split the training and testing videos

    def run(self, img_num):
        validation_set = spatial_dataset(img_num=img_num, root_dir=self.data_path,
                                         transform=transforms.Compose([
                                             transforms.Scale([224, 224]),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])
                                         ]))

        return DataLoader(
            dataset=validation_set,
            batch_size=self.BATCH_SIZE,
            shuffle=False,
            num_workers=self.num_workers)
