import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class loss_whole(nn.Module):
    def __init__(self, hypara):
        super(loss_whole, self).__init__()
        self.std = hypara['W']['W_std']

        self.mask_project = torch.FloatTensor([[0,1,1],[0,1,1],[1,0,1],[1,0,1],[1,1,0],[1,1,0]]).cuda().unsqueeze(0).unsqueeze(0).detach()

        self.mask_plane = torch.FloatTensor([[1,0,0],[1,0,0],[0,1,0],[0,1,0],[0,0,1],[0,0,1]]).cuda().unsqueeze(0).unsqueeze(0).detach()

        self.cube_normal = torch.FloatTensor([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0],[0,0,-1],[0,0,1]]).cuda().unsqueeze(0).unsqueeze(0).detach()
        
        self.cube_planes = torch.FloatTensor([[-1,-1,-1],[1,1,1]]).cuda().unsqueeze(0).unsqueeze(0).detach()

    def compute_REC(self, idx_normals_sim_max, assign_matrix,\
                    scale, pc_inver, pc_sample_inver, planes_scaled, mask_project, mask_plane,\
                    batch_size, num_points, num_cuboids):
        planes_scaled = planes_scaled.unsqueeze(1).repeat(1,num_points,1,3,1).reshape(batch_size,num_points,num_cuboids*6,3)
        scale = scale.unsqueeze(1).repeat(1,num_points,1,6).reshape(batch_size,num_points,num_cuboids*6,3)
        pc_project = pc_sample_inver.permute(0,2,1,3).repeat(1,1,1,6).reshape(batch_size,num_points,num_cuboids*6,3) * mask_project + planes_scaled * mask_plane
        pc_project = torch.max(torch.min(pc_project, scale), -scale).view(batch_size, num_points, num_cuboids, 6, 3)  # [B * num_points * (N*6) * 3]
        pc_project = torch.gather(pc_project, dim=3, index = idx_normals_sim_max.unsqueeze(-1).repeat(1,1,1,1,3)).squeeze(3).permute(0,2,1,3)
        diff = ((pc_project - pc_inver) ** 2).sum(-1).permute(0,2,1)
        diff = torch.mean(torch.mean(torch.sum(diff * assign_matrix, -1), 1))

        return diff

    def compute_SPS(self, assign_matrix):
        num_points = assign_matrix.shape[1]
        norm_05 = (assign_matrix.sum(1)/num_points + 0.01).sqrt().mean(1).pow(2)
        norm_05 = torch.mean(norm_05)

        return norm_05

    def compute_EXT(self, assign_matrix, exist,
                    batch_size, num_points, num_cuboids):
        thred = 24
        loss = nn.BCEWithLogitsLoss().cuda()
        gt = (assign_matrix.sum(1) > thred).to(torch.float32).detach()
        entropy = loss(exist.squeeze(-1), gt)

        return entropy

    def compute_KLD(self, mu , log_var):
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        return kld_loss

    def compute_CST(self, pc_assign_mean, trans):
        diff = torch.norm((pc_assign_mean.detach() - trans), p = 2, dim = -1)
        diff = torch.mean(torch.mean(diff, -1), -1)
        return diff

    def forward(self, pc, normals, out_dict_1, out_dict_2, hypara):
        batch_size = out_dict_1['scale'].shape[0]
        num_cuboids = out_dict_1['scale'].shape[1]
        num_points = pc.shape[1]

        randn_dis = (torch.randn((batch_size,num_points)) * self.std).cuda().detach()
        pc_sample = pc + randn_dis.unsqueeze(-1).repeat(1,1,3) * normals

        pc_sample_inver = pc_sample.unsqueeze(1).repeat(1,num_cuboids,1,1) - out_dict_1['pc_assign_mean'].unsqueeze(2).repeat(1,1,num_points,1)
        pc_sample_inver = torch.einsum('abcd,abde->abce', out_dict_1['rotate'].permute(0,1,3,2), pc_sample_inver.permute(0,1,3,2)).permute(0,1,3,2) #B * N * num_points * 3

        planes_scaled = self.cube_planes.repeat(batch_size,num_cuboids,1,1) * out_dict_1['scale'].unsqueeze(2).repeat(1,1,2,1)

        pc_inver = pc.unsqueeze(1).repeat(1,num_cuboids,1,1) - out_dict_1['pc_assign_mean'].unsqueeze(2).repeat(1,1,num_points,1)
        pc_inver = torch.einsum('abcd,abde->abce', out_dict_1['rotate'].permute(0,1,3,2), pc_inver.permute(0,1,3,2)).permute(0,1,3,2) #B * N * num_points * 3

        normals_inver = normals.unsqueeze(1).repeat(1,num_cuboids,1,1)
        normals_inver = torch.einsum('abcd,abde->abce', out_dict_1['rotate'].permute(0,1,3,2), normals_inver.permute(0,1,3,2)).permute(0,1,3,2) #B * N * num_points * 3

        mask_project = self.mask_project.repeat(batch_size,num_points,num_cuboids,1)
        mask_plane = self.mask_plane.repeat(batch_size,num_points,num_cuboids,1)
        cube_normal = self.cube_normal.unsqueeze(2).repeat(batch_size,num_points,num_cuboids,1,1)

        cos = nn.CosineSimilarity(dim=4, eps=1e-4)
        idx_normals_sim_max = torch.max(cos(normals_inver.permute(0,2,1,3).unsqueeze(3).repeat(1,1,1,6,1),cube_normal),dim=-1,keepdim=True)[1]

        loss_ins = 0
        loss_dict = {}

        # Loss REC
        if hypara['W']['W_REC'] != 0:
            REC = self.compute_REC(idx_normals_sim_max, out_dict_1['assign_matrix'],out_dict_1['scale'],\
                                    pc_inver, pc_sample_inver, planes_scaled, mask_project, mask_plane,\
                                    batch_size, num_points, num_cuboids)
            loss_ins = loss_ins + REC * hypara['W']['W_REC'] 
            loss_dict['REC'] = REC.data.detach().item()

        # Loss SPS
        if hypara['W']['W_SPS']  != 0:
            SPS = self.compute_SPS(out_dict_1['assign_matrix'])
            loss_ins = loss_ins + SPS * hypara['W']['W_SPS']
            loss_dict['SPS'] = SPS.data.detach().item()

        # Loss EXT
        if hypara['W']['W_EXT'] != 0:
            EXT = self.compute_EXT(out_dict_1['assign_matrix'], out_dict_1['exist'],
                                    batch_size, num_points, num_cuboids)
            loss_ins = loss_ins + EXT * hypara['W']['W_EXT']
            loss_dict['EXT'] = EXT.data.detach().item()

        # Loss KLD
        if hypara['W']['W_KLD'] != 0:
            KLD = self.compute_KLD(out_dict_1['mu'],out_dict_1['log_var'])
            loss_ins = loss_ins + KLD * hypara['W']['W_KLD'] 
            loss_dict['KLD'] = KLD.data.detach().item()

        # Loss CST
        if hypara['W']['W_CST'] != 0:
            CST = self.compute_CST(out_dict_1['pc_assign_mean'],out_dict_1['trans'])
            loss_ins = loss_ins + CST * hypara['W']['W_CST']
            loss_dict['CST'] = CST.data.detach().item()

        loss_dict['ALL'] = loss_ins.data.detach().item()

        if hypara['W']['W_SPS']  != 0:
            loss_dict['eval'] = (REC * hypara['W']['W_REC']  + SPS * hypara['W']['W_SPS'] ).data.detach().item()
        else:
            loss_dict['eval'] = (REC * hypara['W']['W_REC']  + 0 * hypara['W']['W_SPS'] ).data.detach().item()
        loss_dict['mu'] = torch.mean(torch.mean(out_dict_1['mu'],1),0).data.detach().item()
        loss_dict['var'] = torch.mean(torch.mean(out_dict_1['log_var'].exp(),1),0).data.detach().item()

        return loss_ins, loss_dict
