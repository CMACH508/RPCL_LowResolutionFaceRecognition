from __future__ import print_function

import torch
import torchvision.transforms as T
import glob
from PIL import Image

torch.backends.cudnn.bencmark = True
import os, sys
import numpy as np
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from os.path import join
from data import Dataset_LFW

base_dir = os.path.abspath('..')
add_path = [base_dir, join(base_dir, 'DUL.cls'), join(base_dir, 'DUL.cls', 'models')]
sys.path += add_path

from models.resnet18 import Resnet18


def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[int(i * n / n_folds):int((i + 1) * n / n_folds)]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds


def eval_acc(threshold, diff, is_print=False):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
    return accuracy


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


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


def get_sample_feature(model, img_paths, batch_size, transforms):
    ss = 0
    flag = True
    first = True
    all_feature = None
    # feature_sum = torch.empty(512).cuda()
    feature_sum = 0
    while flag:
        ''' one batch '''
        if ss + batch_size >= len(img_paths):
            # print(ss, len(img_paths))
            batch_paths = img_paths[ss:len(img_paths)]
            flag = False
        else:
            # print(ss, ss + batch_size)
            batch_paths = img_paths[ss:ss + batch_size]
            ss = ss + batch_size

        batch_input = None
        for p in batch_paths:
            data = Image.open(p)
            if data.mode == 'L':
                data = data.convert("RGB")
            data = transforms(data)
            data = np.asarray(data)
            data = data[np.newaxis, :]
            if batch_input is None:
                batch_input = data
            else:
                batch_input = np.concatenate((batch_input, data), axis=0)

        batch_input = torch.from_numpy(batch_input).float().cuda()

        # print('batch_input=', batch_input.shape)
        batch_features, _, _, _, _ = model(batch_input)
        #
        # batch_features = nn.functional.normalize(batch_features, dim=1)
        # print('batch_features=', batch_features.shape)
        # print(torch.sum(batch_features, dim=0).shape)
        # if first:
        #     feature_sum = torch.sum(batch_features, dim=0)
        #     first = False
        # else:
        #     # feature_sum = feature_sum + torch.sum(batch_features, dim=0)
        #     feature_sum = torch.sum(batch_features, dim=0)
        #     # feature_sum = torch.add(feature_sum, torch.sum(batch_features, dim=0))
        #     # feature_sum = feature_sum + torch.sum(batch_features, dim=0)
        # # print(feature_sum.shape, feature_sum[:5])  # [512]

        if all_feature is None:
            all_feature = batch_features
        else:
            all_feature = torch.cat((all_feature, batch_features), axis=0)
            # print(all_feature.shape)

    # print(feature_sum.shape, feature_sum[:5])

    # feature_avg = feature_sum / len(img_paths)
    # print('get_avg=', feature_avg.shape)
    # feature_avg = nn.functional.normalize(feature_avg, dim=0)
    # return feature_avg
    # return 0
    all_feature = all_feature.data.cpu().numpy()  # (b, 512)
    feature = all_feature / np.linalg.norm(all_feature, axis=1, keepdims=True)  # (b, 512)
    feature_mean = np.mean(feature, axis=0, keepdims=True)  # (1, 512)
    # print(feature_mean.shape)  # (1, 512)
    feature = feature_mean / np.linalg.norm(feature_mean, axis=1)
    feature = np.squeeze(feature)
    # print('final:', feature.shape)

    # feature = nn.functional.normalize(all_feature, dim=1)
    # feature = torch.mean(feature, dim=0)
    # feature = nn.functional.normalize(feature, dim=0)
    return torch.from_numpy(feature).cuda()
    # return 0


def get_sample_paths(test_root, img_dir_1):
    img_paths = glob.glob(join(test_root, img_dir_1, '*'))
    if len(img_paths) % 2 == 1:
        img_paths = img_paths[:-1]
    if len(img_paths) >= 100:
        img_paths = img_paths[:100]
    return img_paths


def eval(model, test_root, test_list, test_batch_size=100):
    print('test_root:', test_root)
    print('test_list:', test_list)
    model.eval()
    predicts = []
    test_batch_size = 50
    with open(os.path.join(test_list), 'r') as fd:
        imgs = fd.readlines()[1:]
    # imgs = imgs[:100]
    len_pair = len(imgs)
    img_pairs = np.random.permutation(imgs)
    transforms = transform(phase='test')
    for i, pair in enumerate(img_pairs):
        splits = pair.replace('\n', '').split(',')
        # print(splits)
        img_dir_1 = splits[2].strip()
        img_dir_2 = splits[3].strip()
        label = np.int8(splits[-1])

        # print('\n{} pair:'.format(i))
        img_paths = get_sample_paths(test_root, img_dir_1)
        # print('1:', len(img_paths))

        f1 = get_sample_feature(model, img_paths, test_batch_size, transforms)

        img_paths = get_sample_paths(test_root, img_dir_2)
        # print('2:', len(img_paths))
        f2 = get_sample_feature(model, img_paths, test_batch_size, transforms)

        cosdistance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)

        predicts.append('{}\t{}\t{}\t{}\n'.format(img_dir_1, img_dir_2, cosdistance, label))
        # print('cos=', cosdistance.item(), 'label=', label)

        if i % 10 == 0:
            print('\n{} pair / {}:'.format(i, len_pair))
            # print('cos=', cosdistance.item(), 'label=', label)

    # config = Config()
    # eval_dataset = Dataset_YTF(test_root, test_list)
    # loader = DataLoader(eval_dataset, shuffle=False, drop_last=False, batch_size=test_batch_size)
    # for ite, (batch_img1, batch_img2, batch_path_1, batch_path_2, is_same) in enumerate(loader):
    #     img1, img2 = batch_img1.numpy(), batch_img2.numpy()
    #     imglist = [img1, img1, img2, img2]
    #     img = np.vstack(imglist)
    #     img = torch.from_numpy(img).float().cuda()
    #
    #     mu, _, _, _, _ = model(img)
    #     out = mu
    #
    #     # std = torch.exp_heatmap(0.5 * logvar)
    #     # eps = torch.randn_like(std)
    #     # out = mu + eps * std
    #
    #     # print('out.shape=', out.shape)
    #     for b in range(test_batch_size):
    #         f1, f2 = out[b], out[b + 2 * test_batch_size]
    #         # print('f1.shape=', f1.shape, 'f2.shape=', f2.shape)
    #         cosdistance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
    #
    #         temp_is_same = is_same[b]
    #         path_1 = batch_path_1[b].split('/')
    #         path_2 = batch_path_2[b].split('/')
    #         if len(path_1) == 1:
    #             temp_path_1 = path_1[-1]
    #             temp_path_2 = path_1[-1]
    #         else:
    #             temp_path_1 = path_1[-2] + path_1[-1]
    #             temp_path_2 = path_2[-2] + path_1[-1]
    #         predicts.append('{}\t{}\t{}\t{}\n'.format(temp_path_1, temp_path_2, cosdistance, temp_is_same))

    accuracy = []
    thd = []
    # folds = KFold(n=5000, n_folds=10, shuffle=False)
    folds = KFold(n=len_pair, n_folds=10, shuffle=False)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    # predicts = np.array(map(lambda line: line.strip('\n').split(), predicts))
    # predicts = np.array(map(lambda line: line.strip('\n').split('-'), predicts))
    predicts_list = []

    # print('len predicts=', len(predicts))
    for line in predicts:
        line = line.strip('\n').split('\t')
        predicts_list.append(line)

    predicts = np.array(predicts_list)

    for idx, (train_index, test_index) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train_index])
        accuracy.append(eval_acc(best_thresh, predicts[test_index]))
        thd.append(best_thresh)
    print('acc_list:', np.round(np.asarray(accuracy), 4))
    print('YTFACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
    return np.mean(accuracy), np.std(accuracy), np.mean(thd)


def eval_lfw(model, test_root, test_list, test_batch_size=100, is_print=False):
    model.eval()
    # config = Config()
    eval_dataset = Dataset_LFW(test_root, test_list)
    loader = DataLoader(eval_dataset, shuffle=False, drop_last=False, batch_size=test_batch_size)
    predicts = []
    for ite, (batch_img1, batch_img2, batch_path_1, batch_path_2, is_same) in enumerate(loader):
        img1, img2 = batch_img1.numpy(), batch_img2.numpy()
        imglist = [img1, img1, img2, img2]
        img = np.vstack(imglist)
        img = torch.from_numpy(img).float().cuda()

        mu, _, _, _, _ = model(img)
        out = mu

        # std = torch.exp_heatmap(0.5 * logvar)
        # eps = torch.randn_like(std)
        # out = mu + eps * std

        # print('out.shape=', out.shape)
        for b in range(test_batch_size):
            f1, f2 = out[b], out[b + 2 * test_batch_size]
            # print('f1.shape=', f1.shape, 'f2.shape=', f2.shape)
            cosdistance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)

            temp_is_same = is_same[b]
            path_1 = batch_path_1[b].split('/')
            path_2 = batch_path_2[b].split('/')
            if len(path_1) == 1:
                temp_path_1 = path_1[-1]
                temp_path_2 = path_1[-1]
            else:
                temp_path_1 = path_1[-2] + '/' + path_1[-1]
                temp_path_2 = path_2[-2] + '/' + path_2[-1]
            predicts.append('{}\t{}\t{}\t{}\n'.format(temp_path_1, temp_path_2, cosdistance, temp_is_same))

    accuracy = []
    thd = []
    folds = KFold(n=6000, n_folds=10, shuffle=False)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    # predicts = np.array(map(lambda line: line.strip('\n').split(), predicts))
    # predicts = np.array(map(lambda line: line.strip('\n').split('-'), predicts))
    predicts_list = []

    # print('len predicts=', len(predicts))
    for line in predicts:
        line = line.strip('\n').split('\t')
        predicts_list.append(line)

    predicts = np.array(predicts_list)

    for idx, (train_index, test_index) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train_index])
        accuracy.append(eval_acc(best_thresh, predicts[test_index], is_print=is_print))
        thd.append(best_thresh)
    print('acc_list:', np.round(np.asarray(accuracy), 4))
    print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
    return np.mean(accuracy), np.std(accuracy), np.mean(thd)


if __name__ == '__main__':
    from DUL.config import Config

    config = Config()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    model = Resnet18(num_classes=10575, embedding_size=512)
    model = DataParallel(model)
    model.to(torch.device("cuda"))

    path = config.test_model_path
    model.load_state_dict(torch.load(path)['backbone'])
    print('load path=', path)
    print('test_root=', config.ytf_test_root)

    eval(model, test_root=config.ytf_test_root, test_list=config.ytf_test_list)
