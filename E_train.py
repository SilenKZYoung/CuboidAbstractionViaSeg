import os
import random
import copy
import argparse
import json
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from data_loader import shapenet4096
from network import Network_Whole
from losses import loss_whole
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

def parsing_hyperparas(args):
    # parsing hyper-parameters to dict
    hypara = {}
    hypara['E'] = {}
    hypara['D'] = {}
    hypara['L'] = {}
    hypara['W'] = {}
    hypara['N'] = {}
    for arg in vars(args):
        hypara[str(arg)[0]][str(arg)] = getattr(args, arg)
    # get the save_path
    save_path = hypara['E']['E_ckpts_folder'] + hypara['E']['E_name'] + '/'
    save_path = save_path + str(hypara['D']['D_datatype'])
    save_path = save_path + '-L'
    for key in hypara['L']:
        save_path = save_path + '_' + str(hypara['L'][key])
    save_path = save_path + '-N'
    for key in hypara['N']:
        save_path = save_path + '_' + str(hypara['N'][key])
    save_path = save_path + '-W'
    for key in hypara['W']:
        save_path = save_path + '_' + str(hypara['W'][key])
    if not os.path.exists(save_path + '/log/'): 
        os.makedirs(save_path + '/log/')
    # save hyper-parameters to json
    with open(save_path + '/hypara.json', 'w') as f:
        json.dump(hypara, f)
    summary_writer = SummaryWriter(save_path + '/tensorboard')

    return hypara, save_path, summary_writer


def main(args):
    hypara, save_path, summary_writer = parsing_hyperparas(args)

    # Choose the CUDA device
    if 'E_CUDA' in hypara['E']:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(hypara['E']['E_CUDA'])

    # Create Dataset
    train_dataset = shapenet4096('train', hypara['E']['E_shapenet4096'], hypara['D']['D_datatype'], True)
    valid_dataset = shapenet4096('valid', hypara['E']['E_shapenet4096'], hypara['D']['D_datatype'], True)

    # Create dataloader
    train_dataloader = DataLoader(train_dataset, 
                                batch_size = hypara['L']['L_batch_size'],
                                shuffle=True, 
                                num_workers=int(hypara['E']['E_workers']), 
                                pin_memory=True)
    val_dataloader = DataLoader(valid_dataset, 
                                batch_size = hypara['L']['L_batch_size'],
                                shuffle=False, 
                                num_workers=int(hypara['E']['E_workers']), 
                                pin_memory=True)
    
    # Create Model
    Network = Network_Whole(hypara).cuda()
    Network.train()

    # Load Model if checkpoint is not none
    if hypara['E']['E_ckpt_path'] != '':
        Network.load_state_dict(torch.load(hypara['E']['E_ckpt_path']))
        print('Load model successfully: %s' % (hypara['E']['E_ckpt_path']))

    # Create Loss Function
    loss_func = loss_whole(hypara).cuda()

    # Create Optimizer
    optimizer = optim.Adam(Network.parameters(), lr = hypara['L']['L_base_lr'], betas = (hypara['L']['L_adam_beta1'], 0.999))
    
    # Training Processing
    best_eval_loss = 100000
    color = utils_pt.generate_ncolors(hypara['N']['N_num_cubes'])
    num_batch = len(train_dataset)/hypara['L']['L_batch_size']
    batch_count = 0
    for epoch in range(hypara['L']['L_epochs']):
        for i, data in enumerate(train_dataloader, 0):
            points, normals, _, _, _ = data
            points, normals = points.cuda(), normals.cuda()
            optimizer.zero_grad()
            outdict = Network(pc = points)
            loss, loss_dict = loss_func(points, normals, outdict, None, hypara)
            loss.backward()
            optimizer.step()
            utils_pt.print_text(loss_dict, save_path, is_train = True, epoch = epoch, i = i, num_batch = num_batch, lr = hypara['L']['L_base_lr'], print_freq_iter = hypara['E']['E_freq_print_iter'])
            batch_count += 1
            if batch_count % int(hypara['E']['E_freq_val_epoch'] * num_batch) == 0:
                utils_pt.train_summaries(summary_writer,loss_dict,batch_count * hypara['L']['L_batch_size'])
                best_eval_loss = validate(hypara, val_dataloader, Network, loss_func, hypara['W'], save_path, batch_count, epoch, summary_writer, best_eval_loss, color)
                Network.train()

def validate(hypara, val_dataloader, Network, loss_func, loss_weight, save_path, iter, epoch, summary_writer, best_eval_loss, color):
    Network.eval()
    loss_dict = {}
    for j, data in enumerate(val_dataloader, 0):
        with torch.no_grad():
            points, normals, _, _, _ = data
            points, normals = points.cuda(), normals.cuda()
            outdict = Network(pc = points)
            _, cur_loss_dict = loss_func(points, normals, outdict, None, hypara)
            if j == 0:
                save_points = points
                save_dict = outdict
            if loss_dict:
                for key in cur_loss_dict:
                    loss_dict[key] = loss_dict[key] + cur_loss_dict[key]
            else:
                loss_dict = cur_loss_dict
    for key in loss_dict:
        loss_dict[key] = loss_dict[key] / (j+1)
        
    utils_pt.print_text(loss_dict, save_path, is_train = False)
    utils_pt.valid_summaries(summary_writer, loss_dict, iter * hypara['L']['L_batch_size'])
    if (loss_dict['eval']) < best_eval_loss:
        best_eval_loss = copy.deepcopy(loss_dict['eval'])
        print('eval: ',best_eval_loss)
        if epoch >= 0:
            model_name = utils_pt.create_name(iter, loss_dict)
            torch.save(Network.state_dict(), save_path +'/'+ model_name + '.pth')
            vertices, faces = utils_pt.generate_cube_mesh_batch(save_dict['verts_forward'], save_dict['cube_face'], hypara['L']['L_batch_size'])
            utils_pt.visualize_segmentation(save_points, color, save_dict['assign_matrix'], save_path + '/log/', 0, None)
            utils_pt.visualize_cubes(vertices, faces, color, save_path + '/log/', 0, '', None)
            utils_pt.visualize_cubes_masked(vertices, faces, color, save_dict['assign_matrix'], save_path + '/log/', 0, '', None)
            vertices_pred, faces_pred = utils_pt.generate_cube_mesh_batch(save_dict['verts_predict'], save_dict['cube_face'], hypara['L']['L_batch_size'])
            utils_pt.visualize_cubes(vertices_pred, faces_pred, color, save_path + '/log/', 0, 'pred', None)
            utils_pt.visualize_cubes_masked(vertices_pred, faces_pred, color, save_dict['assign_matrix'], save_path + '/log/', 0, 'pred', None)
            utils_pt.visualize_cubes_masked_pred(vertices_pred, faces_pred, color, save_dict['exist'], save_path + '/log/', 0, None)
    
    return best_eval_loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Experiment(E) hyper-parameters
    parser.add_argument ('--E_name', default ='EXP_1', type = str, help = 'Experiment name')
    parser.add_argument ('--E_workers', default = 4, type = int, help = 'Number of workers')
    parser.add_argument ('--E_freq_val_epoch', default = 1, type = float, help = 'Frequency of validation')
    parser.add_argument ('--E_freq_print_iter', default = 10, type = int, help = 'Frequency of print')
    parser.add_argument ('--E_CUDA', default = 0, type = int, help = 'Index of CUDA')
    parser.add_argument ('--E_shapenet4096', default = '', type = str, help = 'Path to ShapeNet4096 dataset')
    parser.add_argument ('--E_ckpts_folder', default ='', type = str, help = 'Save path')
    parser.add_argument ('--E_ckpt_path', default ='', type = str, help = '(Optional) Path to checkpoint to load')
    
    # Dataset(D) hyper-parameters
    parser.add_argument ('--D_datatype', default = 'chair', type = str, help = 'airplane, chair, table or animal')
    
    # Learning(L) hyper-parameters
    parser.add_argument ('--L_base_lr', default = 6e-4, type = float, help = 'Learning rate')
    parser.add_argument ('--L_adam_beta1', default = 0.9, type = float, help = 'Adam beta1')
    parser.add_argument ('--L_batch_size', default = 32, type = int, help = 'Batch size')
    parser.add_argument ('--L_epochs', default = 1000, type = int, help = 'Number of epochs')

    # Network(N) hyper-parameters`
    parser.add_argument ('--N_if_low_dim', default = 0, type = int, help = 'DGCNN paramter: KNN manner')
    parser.add_argument ('--N_k', default = 20, type = int, help = 'DGCNN paramter: K of KNN')
    parser.add_argument ('--N_dim_emb', default = 1024, type = int, help = 'Dimension of global feature')
    parser.add_argument ('--N_dim_z', default = 512, type = int, help = 'Dimension of latent code Z')
    parser.add_argument ('--N_dim_att', default = 64, type = int, help = 'Dimension of query and key in attention')
    parser.add_argument ('--N_num_cubes', default = 16, type = int, help = 'Number of cuboids')

    # Weight(W) hyper-parameters of losses
    parser.add_argument ('--W_REC', default = 1.00, type = float, help = 'REC loss weight')
    parser.add_argument ('--W_std', default = 0.05, type = float, help = 'std of normal sampling')
    parser.add_argument ('--W_SPS', default = 0.10, type = float, help = 'SPS loss weight')
    parser.add_argument ('--W_EXT', default = 0.01, type = float, help = 'EXT loss weight')
    parser.add_argument ('--W_KLD', default = 6e-6, type = float, help = 'KLD loss weight')
    parser.add_argument ('--W_CST', default = 0.00, type = float, help = 'CST loss weight, this loss is only for generation application')
    
    args = parser.parse_args()
    main(args)
