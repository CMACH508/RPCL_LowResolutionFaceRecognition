from __future__ import print_function

import torch

torch.backends.cudnn.bencmark = True
import os, sys
import numpy as np
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import torch.nn.functional as F


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
    if accuracy < 0.99 and is_print:
        print('\n-------------------- {} ----------------'.format(accuracy))
        # print(y_true, y_predict)
        print('threshold=', threshold, 'len=', len(y_true))
        cnt = 0
        for k in range(len(y_true)):
            if not y_true[k] == y_predict[k]:
                print('{} {} cos={} pre={}, label={}={}'.format(diff[k][0], diff[k][1], diff[k][2], y_predict[k], diff[k][3], y_true[k]))
                cnt += 1
        print('error=', cnt)
    return accuracy


def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold


def eval(model, test_root, test_list, test_batch_size=100, is_print=False, is_center=False):
    model.eval()
    # config = Config()
    eval_dataset = Dataset_LFW(test_root, test_list)
    loader = DataLoader(eval_dataset, shuffle=False, drop_last=True, batch_size=test_batch_size)
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
            f1, f2 = out[b], out[b+2*test_batch_size]
            # print('f1.shape=', f1.shape, 'f2.shape=', f2.shape)
            if not is_center:
                cosdistance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            else:
                cosdistance = F.pairwise_distance(f1.unsqueeze(0), f2.unsqueeze(0), p=2)[0]

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
    print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
    return np.round(np.mean(accuracy), 4), np.std(accuracy), np.mean(thd)


if __name__ == '__main__':
    from RPCL.config import Config
    config = Config()
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu
    model = Resnet18(num_classes=10575, embedding_size=512)
    model = DataParallel(model)
    model.to(torch.device("cuda"))
    path = config.test_model_path
    model.load_state_dict(torch.load(path)['backbone'])
    print('load path=', path)

    eval(model, test_root=config.test_root, test_list=config.test_list, test_batch_size=20, is_print=False)
    print('model_path=', path)

