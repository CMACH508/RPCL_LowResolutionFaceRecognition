from __future__ import print_function

import torch

torch.backends.cudnn.bencmark = True
import os, sys
import numpy as np
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from os.path import join

base_dir = os.path.abspath('..')
add_path = [base_dir]
sys.path += add_path

from data import Dataset_LFW
from models.resnet18 import Resnet18


def KFold(n=6000, n_folds=10, shuffle=False):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[int(i * n / n_folds):int((i + 1) * n / n_folds)]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds


def eval_acc(threshold, diff):
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


def eval(model, test_root, test_list, batch_size=50):
    model.eval()
    # config = Config()
    eval_dataset = Dataset_LFW(test_root, test_list)
    # eval_dataset = Dataset_CROSS_LFW()
    loader = DataLoader(eval_dataset, shuffle=False, drop_last=False, batch_size=batch_size)
    predicts = []
    for ite, (batch_img1, batch_img2, batch_path_1, batch_path_2, is_same) in enumerate(loader):
        img1, img2 = batch_img1.numpy(), batch_img2.numpy()
        imglist = [img1, img1, img2, img2]
        img = np.vstack(imglist)
        img = torch.from_numpy(img).float().cuda()

        mu, logvar, embedding, logits, _ = model(img)
        out = mu

        # std = torch.exp_heatmap(0.5 * logvar)
        # eps = torch.randn_like(std)
        # out = mu + eps * std

        # print('out.shape=', out.shape)
        for b in range(batch_size):
            f1, f2 = out[b], out[b + 2 * batch_size]
            # print('f1.shape=', f1.shape, 'f2.shape=', f2.shape)

            cosdistance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)

            temp_is_same = is_same[b]
            path_1 = batch_path_1[b].split('/')
            path_2 = batch_path_2[b].split('/')
            temp_path_1 = path_1[-2] + path_1[-1]
            temp_path_2 = path_2[-2] + path_1[-1]
            predicts.append('{}\t{}\t{}\t{}\n'.format(temp_path_1, temp_path_2, cosdistance, temp_is_same))

    accuracy = []
    thd = []
    folds = KFold(n=6000, n_folds=10, shuffle=False)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    # predicts = np.array(map(lambda line: line.strip('\n').split(), predicts))
    # predicts = np.array(map(lambda line: line.strip('\n').split('-'), predicts))
    predicts_list = []

    print('len predicts=', len(predicts))
    for line in predicts:
        line = line.strip('\n').split('\t')
        predicts_list.append(line)

    predicts = np.array(predicts_list)

    for idx, (train_index, test_index) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train_index])
        accuracy.append(eval_acc(best_thresh, predicts[test_index]))
        thd.append(best_thresh)
    print('\nLFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
    print('acc_list:', np.round(np.asarray(accuracy), 4))
    return np.mean(accuracy), np.mean(thd)


if __name__ == '__main__':
    from DUL.config import Config
    config = Config()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    model = Resnet18(num_classes=10575, embedding_size=512)
    model = DataParallel(model)
    model.to(torch.device("cuda"))

    path = config.test_model_path
    model.load_state_dict(torch.load(path)['backbone'])
    print('load successfully,', path)
    eval(model, test_root=config.test_root, test_list=config.test_list, batch_size=20)
    print('load path=', path)
