import os


class Config():
    def __init__(self):
        self.base_dir = os.path.abspath('.')
        self.backbone = 'resnet18'
        # self.gpu = '0, 1'
        # self.gpu = '2, 3'
        self.gpu = '3'
        # self.gpu = '0'

        self.resolution = 16
        ''' arcface, cosface '''
        self.fc_mode = 'cosface'
        self.rival_margin = 0.01

        self.kl_weight = 0.01
        # self.is_rpcl = True
        self.is_rpcl = False
        self.save_dir = os.path.join(self.base_dir, 'cos-rpcl-{}'.format(self.kl_weight))

        self.lr_adjust = [30, 65]
        self.max_epoch = 3
        self.base_lr = 0.01
        self.batch_size = 100
        self.test_batch_size = 10
        self.print_freq = 200
        self.save_freq = 3
        self.weight_decay = 5e-4
        self.BASE_DIR = os.path.abspath('..') + '/'

        ''' train '''
        self.train_root = self.BASE_DIR + 'dataset/training_set/casia/{}x{}_120'.format(self.resolution, self.resolution)
        self.train_list = self.BASE_DIR + 'dataset/training_set/casia/new_cleaned_list.txt'
        # self.test_root = self.BASE_DIR + 'dataset/test_set/LFW/{}x{}_120'.format(self.resolution, self.resolution)
        # self.test_list = self.BASE_DIR + 'dataset/test_set/LFW/lfw_test_pair.txt'

        ''' test '''
        self.test_model_path = self.BASE_DIR + 'checkpoints/high-low-DUL-cos-rpcl-0.01-84.70.pth'

        self.test_root = self.BASE_DIR + 'dataset/test_set/high_low_LFW/16x16_120'
        self.test_list = self.BASE_DIR + 'dataset/test_set/high_low_LFW/lfw_test_pair.txt'

        # self.ytf_test_root = self.BASE_DIR + 'dataset/test_set/YTF/16x16_120'
        # self.ytf_test_list = self.BASE_DIR + 'test_set/YTF/YTF_list.txt'
        self.ytf_test_root = self.BASE_DIR + 'dataset/test_set/high_low_YTF/16x16_120'
        self.ytf_test_list = self.BASE_DIR + 'dataset/test_set/high_low_YTF/YTF_list.txt'
