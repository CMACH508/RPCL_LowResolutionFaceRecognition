from os.path import join
from torch.utils.data import Dataset
import os
from PIL import Image
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


BASE_ROOT = join(os.path.abspath('../'), 'dataset')


class Dataset_CASIA(data.Dataset):
    def __init__(self):
        root = join(BASE_ROOT, 'training_set/casia/16x16_120/')
        data_list_file = join(BASE_ROOT, 'training_set/casia/new_cleaned_list.txt')
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


class Dataset_LFW(Dataset):
    def __init__(self, root=None, data_list_file=None):
        if root is None:
            root = join(BASE_ROOT, 'test_set/LFW/16x16_120')
            data_list_file = join(BASE_ROOT, 'test_set/LFW/lfw_test_pair.txt')
        print('\ntest_root:', root)
        print('test_list:', data_list_file)

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()
        self.root = root
        # imgs = [os.path.join(root, img[:-1]) for img in imgs]
        self.imgs = np.random.permutation(imgs)
        # self.imgs = imgs
        self.transformer = self.transform()
        self.flip_transformer = self.flip_transform()
        self.in_size = (120, 120)

    def __getitem__(self, index):
        return self.get_item(self.imgs, index)

    def __len__(self):
        return len(self.imgs)

    def get_item(self, imgs, index):
        sample = imgs[index]
        splits = sample.split()
        img_path_A = splits[0]
        img_path_B = splits[1]

        data_A1, data_A2 = self.get_one_img(img_path_A)
        data_B1, data_B2 = self.get_one_img(img_path_B)
        label = np.int8(splits[2])
        # print(img_path_A, img_path_B, label)
        return data_A1.float(), data_A2.float(), data_B1.float(), data_B2.float(), img_path_A, img_path_B, label

    def get_one_img(self, img_path):
        data = Image.open(join(self.root, img_path))
        assert data.mode == 'L'
        data = data.convert("RGB")
        if data is None:
            print('img not exist')
            print(join(self.root, img_path))
        data_1 = self.transformer(data)
        data_2 = self.flip_transformer(data)
        return data_1, data_2

    def transform(self):
        normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transforms = T.Compose([
            T.ToTensor(),
            normalize
        ])
        return transforms

    def flip_transform(self):
        normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transforms = T.Compose([
            T.RandomHorizontalFlip(p=1),
            T.ToTensor(),
            normalize
        ])
        return transforms


class train_tinyface(data.Dataset):
    def __init__(self):
        root = join(BASE_ROOT, 'test_set/tinyface/Gray_Training_Set')
        data_list_file = join(BASE_ROOT, 'test_set/tinyface/train_list_file.txt')
        print('train_root:', root)
        print('train_list:', data_list_file)
        # self.is_rgb = is_rgb

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
        # print(img_path, data.size)

        # if not self.is_rgb:
        assert data.mode == 'L'
        data = data.convert("RGB")

        data = transforms(data)
        # print('---', splits)
        label = np.int(splits[1])
        return data.float(), label


class test_tinyface(data.Dataset):
    def __init__(self, data_type, list_type):
        root = join(BASE_ROOT, 'test_set/tinyface/Gray_Testing_Set/{}'.format(data_type))
        data_list_file = join(BASE_ROOT, 'test_set/tinyface/{}'.format(list_type))
        print('root:', root)
        print('list:', data_list_file)
        # self.is_rgb = is_rgb

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()

        self.imgs = [os.path.join(root, img[:-1].split(' ')[-1]) for img in imgs]
        # self.imgs = np.random.permutation(imgs)
        self.in_size = (120, 120)

        print(len(self.imgs))

        self.transforms = transform(phase='test')

    def __getitem__(self, index):
        return self.get_item(self.imgs, index, self.transforms)

    def __len__(self):
        return len(self.imgs)

    def get_item(self, imgs, index, transforms):
        sample = imgs[index]
        # print(sample)
        splits = sample.strip().split(' ')
        img_path = splits[-1]

        data = Image.open(img_path)
        assert data.mode == 'L'
        data = data.convert("RGB")
        data = transforms(data)

        return data.float()

# if __name__ == '__main__':
#     pass
