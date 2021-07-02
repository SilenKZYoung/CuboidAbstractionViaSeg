import os
import random
import json
import numpy as np
import torch
import argparse
from torch.utils.data import DataLoader

from data_loader import shapenet4096
from network import Network_Whole
import utils_pytorch as utils_pt

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
seed = 123
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)            
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed) 

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.E_CUDA)

    if not os.path.exists(args.E_ckpt_path + '/infer/'): 
        os.makedirs(args.E_ckpt_path + '/infer/')

    with open(args.E_ckpt_path + '/infer/' + 'checkpoint.txt', 'w') as f:
        f.write(args.E_ckpt_path + '\n')
        f.write(args.checkpoint)

    with open(args.E_ckpt_path + '/hypara.json') as f:
        hypara = json.load(f) 

    # Create Model
    Network = Network_Whole(hypara).cuda()
    Network.eval()

    # Load Model
    Network.load_state_dict(torch.load(args.E_ckpt_path + '/' + args.checkpoint))
    print('Load model successfully: %s' % (args.E_ckpt_path + '/' + args.checkpoint))

    color = utils_pt.generate_ncolors(hypara['N']['N_num_cubes'])

    hypara['E'] = {}
    hypara['E']['E_shapenet4096'] = args.E_shapenet4096

    # Create Dataset
    batch_size = 32
    if args.infer_test:
        cur_dataset = shapenet4096('test', hypara['E']['E_shapenet4096'], hypara['D']['D_datatype'], True)
        cur_dataloader = DataLoader(cur_dataset, 
                                    batch_size = batch_size,
                                    shuffle=False, 
                                    num_workers=4, 
                                    pin_memory=True)
        infer(args, cur_dataloader, Network, hypara, 'test',batch_size, color)
    if args.infer_train:
        cur_dataset = shapenet4096('train', hypara['E']['E_shapenet4096'], hypara['D']['D_datatype'], True)
        cur_dataloader = DataLoader(cur_dataset, 
                                    batch_size = batch_size,
                                    shuffle=False, 
                                    num_workers=4, 
                                    pin_memory=True)
        infer(args, cur_dataloader, Network, hypara, 'train', batch_size, color)


def infer(args, cur_dataloader, Network, hypara, train_val_test, batch_size, color):
    save_path = args.E_ckpt_path + '/infer/' + train_val_test + '/'
    if not os.path.exists(save_path): 
        os.makedirs(save_path)
    for j, data in enumerate(cur_dataloader, 0):
        with torch.no_grad():
            points, normals, _, _, names = data
            points, normals = points.cuda(), normals.cuda()
            outdict = Network(pc = points)

            vertices, faces = utils_pt.generate_cube_mesh_batch(outdict['verts_forward'], outdict['cube_face'], batch_size)
            utils_pt.visualize_segmentation(points, color, outdict['assign_matrix'], save_path, _, names)
            utils_pt.visualize_cubes(vertices, faces, color, save_path, _, '', names)
            utils_pt.visualize_cubes_masked(vertices, faces, color, outdict['assign_matrix'], save_path, _, '', names)
            vertices_pred, faces_pred = utils_pt.generate_cube_mesh_batch(outdict['verts_predict'], outdict['cube_face'], batch_size)
            utils_pt.visualize_cubes(vertices_pred, faces_pred, color, save_path, _, 'pred', names)
            utils_pt.visualize_cubes_masked(vertices_pred, faces_pred, color, outdict['assign_matrix'], save_path, _, 'pred', names)
            utils_pt.visualize_cubes_masked_pred(vertices_pred, faces_pred, color, outdict['exist'], save_path, _, names)
            print(j)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument ('--E_CUDA', default = 0, type = int, help = 'Index of CUDA')
    parser.add_argument ('--infer_train', default = True, type = bool, help = 'If infer training set')
    parser.add_argument ('--infer_test', default = True, type = bool, help = 'If infer test set')
    
    parser.add_argument ('--E_shapenet4096', default = '', type = str, help = 'Path to ShapeNet4096 dataset')
    parser.add_argument ('--E_ckpt_path', default = '', type = str, help = 'Experiment checkpoint path')
    parser.add_argument ('--checkpoint', default = '', type = str, help = 'Checkpoint name')

    args = parser.parse_args()
    main(args)
