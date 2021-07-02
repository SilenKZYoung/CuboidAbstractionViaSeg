import math
import torch
import numpy as np
import torch.nn as nn
import random
import colorsys
from plyfile import (PlyData, PlyElement)
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def quat2mat(quat):
    B = quat.shape[0]
    N = quat.shape[1]
    quat = quat.contiguous().view(-1,4)
    w, x, y, z = quat[:,0], quat[:,1], quat[:,2], quat[:,3]
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B*N, 3, 3)
    rotMat = rotMat.view(B,N,3,3)
    return rotMat 


def print_text(loss_dict, save_path, is_train, epoch=None, i=None, num_batch=None, lr=None, print_freq_iter = None):
    if is_train:
        if i % print_freq_iter == 0:
            text1 = '[epoch %3d: %3d/%3d] \t lr: %0.6f\t\n' %(epoch+1, i, num_batch, lr)
            text2 = ''
            for item in loss_dict.items():
                text2 = text2 + item[0] + ': %0.6f\t' %(item[1])
            text = text1 + text2 + '\n'
            print(text)
            with open(save_path + '/record.txt',"a") as f:
                f.write(text)
    else:
        text1 = '============== validation time ==============\n'
        text2 = ''
        for item in loss_dict.items():
            text2 = text2 + item[0] + ': %0.6f\t' %(item[1])
        text = text1 + text2 + '\n' + text1
        print(text)
        with open(save_path + '/record.txt',"a") as f:
            f.write(text)


def create_name(iter_times,loss_dict):
    text = ''
    for item in loss_dict.items():
        text = text +'_' +  item[0] + '%0.4f' %(item[1])
    text = 'iter%d'%(iter_times) + text
    return text


def train_summaries(summary_writer,loss_dict,iter_times):
    for item in loss_dict.items():
        summary_writer.add_scalar(item[0], item[1], iter_times)


def valid_summaries(summary_writer,loss_dict,iter_times):
    for item in loss_dict.items():
        summary_writer.add_scalar('eval_' + item[0], item[1], iter_times)


def generate_ncolors(num):
    def get_n_hls_colors(num):
        hls_colors = []
        i = 0
        step = 360.0 / num
        while i < 360:
            h = i
            s = 90 + random.random() * 10
            l = 50 + random.random() * 10
            _hlsc = [h / 360.0, l / 100.0, s / 100.0]
            hls_colors.append(_hlsc)
            i += step
        return hls_colors
    rgb_colors = np.zeros((0,3))
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors = np.concatenate((rgb_colors,np.array([r,g,b])[np.newaxis,:]))
    return rgb_colors


def export_pc(points, colors, filename):
    num_points = points.shape[0]
    if colors is not None:
        vertices = np.empty(num_points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        vertices['x'] = points[:,0].astype('f4')
        vertices['y'] = points[:,1].astype('f4')
        vertices['z'] = points[:,2].astype('f4')
        vertices['red'] = colors[:,0].astype('u1')
        vertices['green'] = colors[:,1].astype('u1')
        vertices['blue'] = colors[:,2].astype('u1')
        ply = PlyData([PlyElement.describe(vertices, 'vertex')], text=False)
        ply.write(filename)
    else:
        vertices = np.empty(num_points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        vertices['x'] = points[:,0].astype('f4')
        vertices['y'] = points[:,1].astype('f4')
        vertices['z'] = points[:,2].astype('f4')
        ply = PlyData([PlyElement.describe(vertices, 'vertex')], text=False)
        ply.write(filename)


def export_mesh(vertices, faces_idx, vertex_color, face_color, filename):
    if vertex_color is not None:
        vertex = np.zeros(vertices.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
        for i in range(vertices.shape[0]):
            vertex[i] = (vertices[i][0], vertices[i][1], vertices[i][2],vertex_color[i,0],vertex_color[i,1],vertex_color[i,2])
    else:
        vertex = np.zeros(vertices.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        for i in range(vertices.shape[0]):
            vertex[i] = (vertices[i][0], vertices[i][1], vertices[i][2])
    if face_color is not None:
        faces = np.zeros(faces_idx.shape[0], dtype=[('vertex_indices', 'i4', (3,)),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
        for i in range(faces_idx.shape[0]):
            faces[i] = ([faces_idx[i][0], faces_idx[i][1], faces_idx[i][2]],face_color[i,0],face_color[i,1],face_color[i,2])
    else:
        faces = np.zeros(faces_idx.shape[0], dtype=[('vertex_indices', 'i4', (3,))])
        for i in range(faces_idx.shape[0]):
            faces[i] = ([faces_idx[i][0], faces_idx[i][1], faces_idx[i][2]])

    ply_out = PlyData([PlyElement.describe(vertex, 'vertex', comments=['vertices']),
                       PlyElement.describe(faces, 'face')],text=True)
    ply_out.write(filename)


def generate_cube_mesh_batch(vertices, cube_face, num_samples):
    batch_size = vertices.shape[0]
    num_cuboids = vertices.shape[1]
    vertices = vertices.reshape(batch_size,-1,3).detach()  
    faces_one_cube = cube_face.unsqueeze(0).repeat(batch_size,1,1).float().detach() 
    faces = torch.zeros((batch_size,0,3)).float().cuda()
    for i in range(0,num_cuboids):
        faces = torch.cat((faces, faces_one_cube + i*8),1)
    faces = faces.int()

    return vertices, faces

def visualize_cubes(vertices, faces, color, save_path, bias, pred, names):
    batch_size = vertices.shape[0]
    vertices = vertices.cpu().numpy()
    faces = faces.cpu().numpy()
    for k in range(batch_size):
        if names is not None:
            filename = save_path + names[k] + '_cube' + pred + '.ply'
        else:
            filename = save_path + str(bias + k) + '_cube' + pred + '.ply'
        faces_color = color[:,np.newaxis,:].repeat(24,axis = 1).reshape(-1,3)
        vertex_color = color[:,np.newaxis,:].repeat(8,axis = 1).reshape(-1,3)
        export_mesh(vertices[k,:,:], faces[k,:,:],vertex_color, faces_color, filename)


def visualize_cubes_masked(vertices, faces, color, assign_matrix, save_path, bias, pred, names):
    threhold = 24
    batch_size = assign_matrix.shape[0]
    num_cuboids = assign_matrix.shape[2]
    vertices = vertices.cpu().numpy()
    faces = faces.cpu().numpy()
    assign_matrix = assign_matrix.cpu().numpy()
    for k in range(batch_size):
        if names is not None:
            filename = save_path + names[k] + '_cube_masked' + pred + '.ply'
        else:
            filename = save_path + str(bias + k) + '_cube_masked' + pred + '.ply'
        mask = assign_matrix[k,:,:].sum(0)>threhold
        vertices_masked = vertices[k,:,:].reshape(num_cuboids,8,3)[mask,:,:].reshape(-1,3)
        faces_masked = faces[k,:mask.sum()*24,:]
        faces_color = color[mask,:][:,np.newaxis,:].repeat(24,axis = 1).reshape(-1,3)
        vertex_color = color[mask,:][:,np.newaxis,:].repeat(8,axis = 1).reshape(-1,3)
        export_mesh(vertices_masked, faces_masked, vertex_color, faces_color, filename)


def visualize_cubes_masked_pred(vertices, faces, color, pred_mask, save_path, bias, names):
    batch_size = pred_mask.shape[0]
    num_cuboids = pred_mask.shape[1]
    func_Sigmoid = nn.Sigmoid()
    vertices = vertices.cpu().numpy()
    faces = faces.cpu().numpy()
    pred_mask = func_Sigmoid(pred_mask.squeeze(-1)).detach().cpu().numpy()
    for k in range(batch_size):
        if names is not None:
            filename = save_path + names[k] + '_cube_masked_all_pred.ply'
        else:
            filename = save_path + str(bias + k) + '_cube_masked_all_pred' + '.ply'
        mask = pred_mask[k,:]>0.5
        vertices_masked = vertices[k,:,:].reshape(num_cuboids,8,3)[mask,:,:].reshape(-1,3)
        faces_masked = faces[k,:mask.sum()*24,:]
        faces_color = color[mask,:][:,np.newaxis,:].repeat(24,axis = 1).reshape(-1,3)
        vertex_color = color[mask,:][:,np.newaxis,:].repeat(8,axis = 1).reshape(-1,3)
        export_mesh(vertices_masked, faces_masked, vertex_color, faces_color, filename)


def visualize_segmentation(pc, color, assign_matrix, save_path, bias, names):
    batch_size = pc.shape[0]
    pc = pc.cpu().detach().numpy()
    assign_matrix = assign_matrix.cpu().numpy()
    for k in range(batch_size):
        if names is not None:
            filename = save_path + names[k] + '_segment.ply'
        else:
            filename = save_path + str(bias + k) + '_segment.ply'
        color_segment = color[np.argmax(assign_matrix[k,:,:],1),:]
        export_pc(pc[k,:,:], color_segment, filename)

