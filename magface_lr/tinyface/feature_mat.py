# _date_:2021/11/4 14:58

import os
import sys, argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
import datetime
from torch.nn import DataParallel
from torch import nn as nn
import scipy.io as scio

torch.backends.cudnn.bencmark = True
from os.path import join

base_dir = os.path.abspath('../..')
add_path = [base_dir, join(base_dir, 'magface_lr')]
sys.path += add_path

from magface_lr.magface import SoftmaxBuilder
from magface_lr.data import test_tinyface
from RPCL.config import Config

config = Config()
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='')
parser.add_argument('--batch_size', type=int, default=100, help='')
parser.add_argument('--is_rpcl', type=int, default=1, help='')
parser.add_argument('--f', type=float, default=0.5, help='')
parser.add_argument('--thred', type=float, default=0.7, help='')
parser.add_argument('--ckpt_path', type=str, default=config.test_model_path, help='')
args = parser.parse_args()
print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def test():
    device = torch.device("cuda:0")
    device_ids = list(range(torch.cuda.device_count()))
    # print(device_ids)
    num_class = 2570
    model = SoftmaxBuilder(class_num=num_class, is_rpcl=args.is_rpcl, thred=args.thred, f=args.f)
    # model = Resnet18(num_classes=6609)
    model = DataParallel(model, device_ids=device_ids).to(device)
    model.train()

    batch_size = args.batch_size
    ckpt_path = args.ckpt_path

    a = os.path.dirname(ckpt_path)
    b = os.path.basename(ckpt_path)
    feat_path = join(a, b+'_feature')
    model.load_state_dict(torch.load(ckpt_path)['model'])
    print('------------------ load model successfully -----------------')

    if not os.path.exists(feat_path):
        os.mkdir(feat_path)

    print('ckpt_path=', ckpt_path)
    print('feat_path=', feat_path)

    dir_list = ['Probe', 'Gallery_Match', 'Gallery_Distractor']
    img_list = ['probe_img_ID_pairs.txt', 'gallery_match_img_ID_pairs.txt', 'distractor_names.txt']

    save_name_list = ['probe.mat', 'gallery.mat', 'distractor.mat']
    map_list = ['probe_feature_map', 'gallery_feature_map', 'distractor_feature_map']

    model.eval()
    with torch.no_grad():
        for i in range(3):
            dataset = test_tinyface(dir_list[i], img_list[i])

            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=6)

            save_data = None
            length = len(dataloader)
            for ite, (img) in enumerate(dataloader):
                # if ite == 10 and i == 2:
                #     break

                if ite % 10 == 0:
                    print('[{:.2f}%]'.format(ite * 100 / length))

                img = img.to(device)

                mu = model(img, is_test=True)
                mu = nn.functional.normalize(mu, dim=1)
                mu = mu.data.cpu().numpy()

                # print(ite, mu.shape)
                if save_data is None:
                    save_data = mu
                else:
                    save_data = np.concatenate((save_data, mu), axis=0)

            scio.savemat(os.path.join(feat_path, save_name_list[i]), {map_list[i]: save_data})

    print('ckpt_path=', ckpt_path)
    print('feature_path=', feat_path)
    print('---- finished ----')


if __name__ == '__main__':
    test()
