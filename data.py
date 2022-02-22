import os
import cv2
from os.path import join
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import glob


def transform(phase):
    normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    if phase == 'train':
        transforms = T.Compose([
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize
        ])
    else:
        transforms = T.Compose([
            T.ToTensor(),
            normalize
        ])
    return transforms


class Dataset_CASIA(data.Dataset):
    def __init__(self, root, data_list_file):
        print('train_root:', root)
        print('train_list:', data_list_file)

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()

        imgs = [os.path.join(root, img[:-1]) for img in imgs]
        self.imgs = np.random.permutation(imgs)
        self.in_size = (120, 120)

        print(len(self.imgs))

        self.transforms = transform(phase='train')

    def __getitem__(self, index):
        return self.get_item(self.imgs, index, self.transforms)

    def __len__(self):
        return len(self.imgs)

    def get_item(self, imgs, index, transforms):
        sample = imgs[index]
        splits = sample.strip().split(' ')
        img_path = splits[0]

        data = Image.open(img_path)
        assert data.mode == 'L'
        data = data.convert("RGB")
        # if data.mode == 'RGB':
        #     data = data.convert("L")
        #     data = data.convert("RGB")
        #
        # elif data.mode == 'L':
        #     data = data.convert("RGB")
        # data = cv2.imread(img_path)[:, :, ::-1]
        # if not data.shape[0:2] == self.in_size:
        #     data = cv2.resize(data, self.in_size)
        # data = Image.fromarray(data)
        data = transforms(data)
        # print('---', splits)
        label = np.int(splits[1])
        return data.float(), label


class Dataset_CASIA_HR_LR(data.Dataset):
    def __init__(self, lr_root, hr_root, data_list_file):
        print('train_root:', lr_root, hr_root)
        print('train_list:', data_list_file)
        self.lr_root = lr_root
        self.hr_root = hr_root
        self.data_list_file = data_list_file

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()

        imgs = [img[:-1] for img in imgs]
        self.imgs = np.random.permutation(imgs)
        self.in_size = (120, 120)

        self.transforms = transform(phase='train')

    def __getitem__(self, index):
        lr, label1 = self.get_item(self.lr_root, self.imgs, index, self.transforms)
        hr, label2 = self.get_item(self.hr_root, self.imgs, index, self.transforms)
        assert label1 == label2
        return lr, hr, label1

    def __len__(self):
        return len(self.imgs)

    def get_item(self, root, imgs, index, transforms):
        sample = imgs[index]
        splits = sample.split()
        img_path = splits[0]

        data = Image.open(join(root, img_path))
        assert data.mode == 'L'
        data = data.convert("RGB")
        # if data.mode == 'RGB':
        #     data = data.convert("L")
        #     data = data.convert("RGB")
        #
        # elif data.mode == 'L':
        #     data = data.convert("RGB")
        # data = cv2.imread(img_path)[:, :, ::-1]
        # if not data.shape[0:2] == self.in_size:
        #     data = cv2.resize(data, self.in_size)
        # data = Image.fromarray(data)
        data = transforms(data)
        label = np.int(splits[1])
        return data.float(), label

class Dataset_LFW(data.Dataset):
    def __init__(self, root, data_list_file, is_gray=True):
        print('\ntest_root:', root)
        print('test_list:', data_list_file)

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()
        self.root = root
        # imgs = [os.path.join(root, img[:-1]) for img in imgs]
        self.imgs = np.random.permutation(imgs)
        # self.imgs = imgs
        self.transforms = transform(phase='test')
        self.in_size = (120, 120)

    def __getitem__(self, index):
        return self.get_item(self.imgs, index, self.transforms)

    def __len__(self):
        return len(self.imgs)

    def get_item(self, imgs, index, transforms):
        sample = imgs[index]
        splits = sample.split()
        img_path_1 = splits[0]
        img_path_2 = splits[1]

        data_1 = Image.open(join(self.root, img_path_1))
        assert data_1.mode == 'L'
        data_1 = data_1.convert("RGB")
        # if data_1.mode == 'RGB':
        #     data_1 = data_1.convert("L")
        #     data_1 = data_1.convert("RGB")
        # elif data_1.mode == 'L':
        #     data_1 = data_1.convert("RGB")

        data_2 = Image.open(join(self.root, img_path_2))
        assert data_2.mode == 'L'
        data_2 = data_2.convert("RGB")
        # if data_2.mode == 'RGB':
        #     data_2 = data_2.convert("L")
        #     data_2 = data_2.convert("RGB")
        # elif data_2.mode == 'L':
        #     data_2 = data_2.convert("RGB")

        # data_1 = cv2.resize(cv2.imread(join(self.root, img_path_1))[:, :, ::-1], self.in_size)  # TODO img = cv2.imread(join(self.root, img_path_1))
        # data_2 = cv2.resize(cv2.imread(join(self.root, img_path_2))[:, :, ::-1], self.in_size)
        # data_1 = Image.fromarray(data_1)
        # data_2 = Image.fromarray(data_2)

        if data_1 is None or data_2 is None:
            print('img not exist')
            print(join(self.root, img_path_1))
            print(join(self.root, img_path_2))
        data_1 = transforms(data_1)
        data_2 = transforms(data_2)
        label = np.int8(splits[2])
        return data_1.float(), data_2.float(), img_path_1, img_path_2, label


class Dataset_CALFW(data.Dataset):
    def __init__(self, root, data_list_file):
        print('\nCALFW test_root:', root)
        print('CALFW test_list:', data_list_file)

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()
        self.root = root
        # imgs = [os.path.join(root, img[:-1]) for img in imgs]
        self.imgs = np.random.permutation(imgs)
        self.transforms = transform(phase='test')
        self.in_size = (120, 120)

    def __getitem__(self, index):
        return self.get_item(self.imgs, index, self.transforms)

    def __len__(self):
        return len(self.imgs)

    def get_item(self, imgs, index, transforms):
        sample = imgs[index]
        splits = sample.split()
        img_path_1 = splits[0]
        img_path_2 = splits[1]
        data_1 = Image.open(join(self.root, img_path_1))
        if data_1.mode == 'L':
            data_1 = data_1.convert("RGB")
        data_2 = Image.open(join(self.root, img_path_2))
        if data_2.mode == 'L':
            data_2 = data_2.convert("RGB")
        # data_1 = cv2.resize(cv2.imread(join(self.root, img_path_1))[:, :, ::-1], self.in_size)  # TODO img = cv2.imread(join(self.root, img_path_1))
        # data_2 = cv2.resize(cv2.imread(join(self.root, img_path_2))[:, :, ::-1], self.in_size)
        # data_1 = Image.fromarray(data_1)
        # data_2 = Image.fromarray(data_2)

        if data_1 is None or data_2 is None:
            print('img not exist')
            print(join(self.root, img_path_1))
            print(join(self.root, img_path_2))
        data_1 = transforms(data_1)
        data_2 = transforms(data_2)
        if not splits[2] == '0':
            splits[2] = 1
        else:
            splits[2] = 0
        label = np.int8(splits[2])
        return data_1.float(), data_2.float(), img_path_1, img_path_2, label


if __name__ == '__main__':
    calfw_root = '/home/lipeiying/program/_FaceRecognition_/dataset/CALFW/CALFW_{}to120'.format(16)
    calfw_list = '/home/lipeiying/program/_FaceRecognition_/dataset/CALFW/pairs_calfw.txt'
    calfw = Dataset_CALFW(calfw_root, calfw_list)
    loader = data.DataLoader(calfw, batch_size=1)
    cnt = 0
    for (data_1, data_2, img_path_1, img_path_2, label) in loader:
        cnt += 1
        print('\npair--')
        print(img_path_1[0])
        print(img_path_2[0])
        print(label[0])
        if cnt == 10:
            break


class Dataset_YTF(data.Dataset):
    def __init__(self, root, data_list_file):
        print('test_root:', root)
        print('test_list:', data_list_file)

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()[1:]
        self.root = root
        # imgs = [os.path.join(root, img[:-1]) for img in imgs]
        self.imgs = np.random.permutation(imgs)
        self.transforms = transform(phase='test')
        # self.in_size = (120, 120)

    def __getitem__(self, index):
        h, l = int(index / 2), index % 2
        sample = self.imgs[h]
        splits = sample.split(',')
        img_dir = splits[2 + l].strip()

        label = np.int8(splits[-1])

        return self.get_item(img_dir, self.transforms, label)

    def __len__(self):
        return len(self.imgs) * 2

    def get_item(self, img_dir, transforms, label):

        img_paths = glob.glob(join(self.root, img_dir, '*'))
        data_list = []
        for img in img_paths:
            data = Image.open(img)
            if data.mode == 'L':
                data = data.convert("RGB")
            data = transforms(data)
            data_list.append(data.float())

        return data_list, img_dir, label

        # img_path_1 = glob.glob(join(self.root, img_dir_1, '*'))
        # img_path_1A, img_path_1B = img_path_1[0], img_path_1[1]
        # img_path_2 = glob.glob(join(self.root, img_dir_2, '*'))
        # img_path_2A, img_path_2B = img_path_2[0], img_path_2[1]
        #
        # data_1A = Image.open(img_path_1A)
        # data_1B = Image.open(img_path_1B)
        # if data_1A.mode == 'L':
        #     data_1A = data_1A.convert("RGB")
        #     data_1B = data_1B.convert("RGB")
        # data_2A = Image.open(img_path_2A)
        # data_2B = Image.open(img_path_2B)
        # if data_2A.mode == 'L':
        #     data_2A = data_2A.convert("RGB")
        #     data_2B = data_2B.convert("RGB")
        # # data_1 = cv2.resize(cv2.imread(join(self.root, img_path_1))[:, :, ::-1], self.in_size)  # TODO img = cv2.imread(join(self.root, img_path_1))
        # # data_2 = cv2.resize(cv2.imread(join(self.root, img_path_2))[:, :, ::-1], self.in_size)
        # # data_1 = Image.fromarray(data_1)
        # # data_2 = Image.fromarray(data_2)
        #
        # if data_1A is None or data_2A is None:
        #     print('img not exist')
        #     print(join(self.root, img_path_1))
        #     print(join(self.root, img_path_2))
        # data_1A = transforms(data_1A)
        # data_1B = transforms(data_1B)
        # data_2A = transforms(data_2A)
        # data_2B = transforms(data_2B)
        # label = np.int8(splits[-1])
        # return data_1A.float(), data_1B.float(), data_2A.float(), data_2B.float(), img_path_1A, img_path_2A, label


import random


class Dataset_CASIA_NOISE(data.Dataset):
    def __init__(self, root, data_list_file):
        print('train_root:', root)
        print('train_list:', data_list_file)

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()

        imgs = [os.path.join(root, img[:-1]) for img in imgs]
        self.imgs = np.random.permutation(imgs)
        self.in_size = (120, 120)

        self.transforms = transform(phase='train')

    def __getitem__(self, index):
        return self.get_item(self.imgs, index, self.transforms)

    def __len__(self):
        return len(self.imgs)

    def get_item(self, imgs, index, transforms):
        sample = imgs[index]
        splits = sample.split()
        img_path = splits[0]
        # data = Image.open(img_path)
        # if data.mode == 'L':
        #     data = data.convert("RGB")
        data = cv2.resize(cv2.imread(img_path)[:, :, ::-1], self.in_size)

        kernel = random.choice([1, 3, 5, 7, 9])  # 越大越模糊 正数和奇数
        data = cv2.GaussianBlur(data, (kernel, kernel), 0)
        data = Image.fromarray(data)

        data = transforms(data)
        label = np.int(splits[1])
        return data.float(), label


class Dataset_CROSS_LFW(data.Dataset):
    def __init__(self):
        root = '/home/lipeiying/program/_FaceRecognition_/dataset/LabeledFacesintheWildHome'
        data_list_file = '/home/lipeiying/program/_FaceRecognition_/dataset/LabeledFacesintheWildHome/CR_LFW_list.txt'
        print('test_root:', root)
        print('test_list:', data_list_file)

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()
        self.root = root
        # imgs = [os.path.join(root, img[:-1]) for img in imgs]
        self.imgs = np.random.permutation(imgs)
        self.transforms = transform(phase='test')
        self.in_size = (120, 120)

    def __getitem__(self, index):
        return self.get_item(self.imgs, index, self.transforms)

    def __len__(self):
        return len(self.imgs)

    def get_item(self, imgs, index, transforms):
        sample = imgs[index]
        splits = sample.split()
        img_path_1 = splits[0]
        img_path_2 = splits[1]
        img_path_1 = img_path_1.replace('mtcnn', '120_mtcnn')
        img_path_2 = img_path_2.replace('mtcnn', '120_mtcnn')
        data_1 = Image.open(join(self.root, img_path_1))
        if data_1.mode == 'L':
            data_1 = data_1.convert("RGB")
        data_2 = Image.open(join(self.root, img_path_2))
        if data_2.mode == 'L':
            data_2 = data_2.convert("RGB")
        # data_1 = cv2.resize(cv2.imread(join(self.root, img_path_1)), self.in_size)  # TODO img = cv2.imread(join(self.root, img_path_1))
        # data_2 = cv2.resize(cv2.imread(join(self.root, img_path_2)), self.in_size)
        # data = data.convert('L')
        if data_1 is None or data_2 is None:
            print('img not exist')
            print(join(self.root, img_path_1))
            print(join(self.root, img_path_2))
        data_1 = transforms(data_1)
        data_2 = transforms(data_2)
        label = np.int8(splits[2])
        return data_1.float(), data_2.float(), img_path_1, img_path_2, label
