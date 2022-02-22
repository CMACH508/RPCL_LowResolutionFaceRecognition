import os


class Config():
    def __init__(self):
        self.base_dir = os.path.abspath('.')

        # self.gpu = '0, 1'
        self.gpu = '2, 3'
        self.resolution = 16
        self.fc_mode = 'cosface'
        # self.rival_margin = 0.2
        # self.rival_margin = 0.05
        self.backbone = 'resnet18'
        self.is_share = False

        # self.is_rival = False
        self.is_rival = True

        self.save_dir = os.path.join(self.base_dir, '{}-cos-rpcl'.format(self.backbone))

        # self.lr_adjust = [10, 35]
        self.lr_adjust = [16, 30, 35]
        # self.lr_adjust = [18]
        self.max_epoch = 3
        self.base_lr = 0.01
        self.batch_size = 100
        self.test_batch_size = 20
        self.print_freq = 150
        self.save_freq = 3
        self.weight_decay = 5e-4

        self.BASE_DIR = os.path.abspath('..') + '/'
        ''' train setting '''
        self.train_root = self.BASE_DIR + 'dataset/training_set/casia/{}x{}_120'.format(self.resolution, self.resolution)
        self.train_list = self.BASE_DIR + 'dataset/training_set/casia/new_cleaned_list.txt'
        # self.train_root = self.BASE_DIR + 'dataset/training_set/high_low_casia/{}x{}_120'.format(self.resolution, self.resolution)
        # self.train_list = self.BASE_DIR + 'dataset/training_set/high_low_casia/new_cleaned_list.txt'

        ''' test setting '''
        # self.test_model_path = self.BASE_DIR + 'checkpoints/RPCL-Cos_lfw_0.9513sota.pth'
        # self.test_model_path = self.BASE_DIR + 'checkpoints/high-low-se-cos-rpcl.pth'
        self.test_model_path = self.BASE_DIR + 'checkpoints/rpcl_magface_sota_f05t07.pth'

        # self.test_root = self.BASE_DIR + 'dataset/test_set/high_low_LFW/{}x{}_120'.format(self.resolution, self.resolution)
        # self.test_list = self.BASE_DIR + 'dataset/test_set/high_low_LFW/lfw_test_pair.txt'
        self.test_root = self.BASE_DIR + 'dataset/test_set/LFW/{}x{}_120'.format(self.resolution, self.resolution)
        self.test_list = self.BASE_DIR + 'dataset/test_set/LFW/lfw_test_pair.txt'

        # self.ytf_test_root = self.BASE_DIR + 'dataset/test_set/YTF/16x16_120'
        # self.ytf_test_list = self.BASE_DIR + 'test_set/YTF/YTF_list.txt'
        self.ytf_test_root = self.BASE_DIR + 'dataset/test_set/high_low_YTF/16x16_120'
        self.ytf_test_list = self.BASE_DIR + 'dataset/test_set/high_low_YTF/YTF_list.txt'
