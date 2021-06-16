import os, sys, h5py
import numpy as np
import torch
import torch.utils.data as data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

class shapenet4096(data.Dataset):
    def __init__(self, phase, data_root, data_type, if_4096):
        super().__init__()
        self.folder = data_type + '/'
        if phase == 'train':
            self.data_list_file = data_root + data_type + '_train.npy'
        else:
            self.data_list_file = data_root + data_type + '_test.npy'
        self.data_dir = data_root + self.folder
        self.data_list = np.load(self.data_list_file)
        
    def __getitem__(self, idx):
        cur_name = self.data_list[idx].split('.')[0]
        cur_data = torch.from_numpy(np.load(self.data_dir + self.data_list[idx])).float()
        cur_points = cur_data[:,0:3]
        cur_normals = cur_data[:,3:]
        cur_points_num = 4096
        cur_values = -1
        return cur_points, cur_normals, cur_points_num, cur_values, cur_name
        
    def __len__(self):
        return self.data_list.shape[0]

