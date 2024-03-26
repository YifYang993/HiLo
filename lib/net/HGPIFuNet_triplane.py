# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
import sys 
# from torchvision.utils import save_image
sys.path.append("/mnt/cephfs/home/yangyifan/yangyifan/code/avatar/ICON/")
from lib.net.voxelize import Voxelization
from lib.dataset.mesh_util import feat_select, read_smpl_constants
from lib.net.NormalNet import NormalNet
from lib.net.MLP import MLP
from lib.net.MLP3D import MLP3d
from lib.net.MLP_N_shape import MLP_UNET
from lib.net.spatial import SpatialEncoder
from lib.dataset.PointFeat import PointFeat
from lib.dataset.mesh_util import SMPLX
from lib.net.VE import VolumeEncoder
from lib.net.HGFilters import *
from termcolor import colored
from lib.net.BasePIFuNet import BasePIFuNet
import torch.nn as nn
import torch
# import time
import torch.nn.functional as F
import torch
import math

def soft_assignment(x,y,weights):
    miu = torch.mean(x)
    sigma = torch.std(x)
    x_hat = miu + weights*sigma
    x_hat = x_hat.repeat(x.size(0),1)
    x = x[:,None]
    y = y[:,None]
    x_dist = 10*torch.exp(-(x_hat-x)**2) #/5
    x_p = F.softmax(x_dist,dim=1)
    x_p = torch.sum(x_p,dim=0)/len(x)
    y_dist = 10 * torch.exp(-(x_hat - y) ** 2) #/ 5
    y_p = F.softmax(y_dist, dim=1)
    y_p = torch.sum(y_p,dim=0)/len(x)
    return x_p,y_p


class HGPIFuNet(BasePIFuNet):
    """
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    """
    def __init__(self, cfg, projection_mode="orthogonal", error_term=nn.MSELoss(), args=None):

        super(HGPIFuNet, self).__init__(projection_mode=projection_mode, error_term=error_term)
        self.cfg=cfg
        self.args=args
        self.l1_loss = nn.SmoothL1Loss()
        self.opt = cfg.net
        self.root = cfg.root
        self.overfit = cfg.overfit

        channels_IF = self.opt.mlp_dim

        self.use_filter = self.opt.use_filter
        self.prior_type = self.opt.prior_type
        self.smpl_feats = self.opt.smpl_feats

        self.smpl_dim = self.opt.smpl_dim
        self.voxel_dim = self.opt.voxel_dim
        self.hourglass_dim = self.opt.hourglass_dim

        self.in_geo = [item[0] for item in self.opt.in_geo]
        self.in_nml = [item[0] for item in self.opt.in_nml]

        self.in_geo_dim = sum([item[1] for item in self.opt.in_geo])
        self.in_nml_dim = sum([item[1] for item in self.opt.in_nml])

        self.in_total = self.in_geo + self.in_nml
        self.smpl_feat_dict = None
        self.smplx_data = SMPLX()

        image_lst = [0, 1, 2]
        normal_F_lst = [0, 1, 2] if "image" not in self.in_geo else [3, 4, 5]
        normal_B_lst = [3, 4, 5] if "image" not in self.in_geo else [6, 7, 8]

        # only ICON or ICON-Keypoint use visibility

        if self.prior_type in ["icon", "keypoint"]:
            if "image" in self.in_geo:
                self.channels_filter = [
                    image_lst + normal_F_lst,
                    image_lst + normal_B_lst,
                ]
            else:
                self.channels_filter = [normal_F_lst, normal_B_lst]

        else:
            # if "image" in self.in_geo and not cfg.exclude_normal:
            #     self.channels_filter = [image_lst + normal_F_lst + normal_B_lst]
            # elif "image" not in self.in_geo:
            #     self.channels_filter = [normal_F_lst + normal_B_lst]
            # elif "image" in self.in_geo and cfg.exclude_normal:
            #     self.channels_filter = [image_lst]
            if "image" in self.in_geo:
                self.channels_filter = [image_lst] 
            if "normal_F" in self.in_geo:
                self.channels_filter += [normal_F_lst] 
            if "normal_F" in self.in_geo:
                self.channels_filter += [normal_B_lst] 


        use_vis = (self.prior_type in ["icon", "keypoint"]) and ("vis" in self.smpl_feats)
        if self.prior_type in ["pamir", "pifu"]:
            use_vis = 1

        if self.use_filter:
            channels_IF[0] = (self.hourglass_dim) * (2 - use_vis)
        else:
            channels_IF[0] = len(self.channels_filter[0]) * (2 - use_vis)

        if self.prior_type in ["icon", "keypoint"]:
            channels_IF[0] += self.smpl_dim
        elif self.prior_type == "pamir":
            channels_IF[0] += self.voxel_dim
            (
                smpl_vertex_code,
                smpl_face_code,
                smpl_faces,
                smpl_tetras,
            ) = read_smpl_constants(self.smplx_data.tedra_dir)
            self.voxelization = Voxelization(
                smpl_vertex_code,
                smpl_face_code,
                smpl_faces,
                smpl_tetras,
                volume_res=128,
                sigma=0.05,
                smooth_kernel_size=7,
                batch_size=self.cfg.batch_size,
                device=torch.device(f"cuda:{self.cfg.gpus[0]}"),
            )
            self.ve = VolumeEncoder(3, self.voxel_dim, self.opt.num_stack)

        elif self.prior_type == "pifu":
            channels_IF[0] += 1
        else:
            print(f"don't support {self.prior_type}!")

        self.base_keys = ["smpl_verts", "smpl_faces"]

        self.icon_keys = self.base_keys + [f"smpl_{feat_name}" for feat_name in self.smpl_feats]
        self.keypoint_keys = self.base_keys + [f"smpl_{feat_name}" for feat_name in self.smpl_feats]

        self.pamir_keys = ["voxel_verts", "voxel_faces", "pad_v_num", "pad_f_num"]
        self.pifu_keys = []
        if args.mlp3d:
            self.if_regressor = MLP3d(
                filter_channels=channels_IF,
                name="if",
                res_layers=self.opt.res_layers,
                norm=self.opt.norm_mlp,
                last_op=nn.Sigmoid() if not self.cfg.test_mode else None,
                args=args
            )
        elif args.use_unet:
            self.if_regressor = MLP_UNET(
                filter_channels=channels_IF,
                name="if",
                res_layers=self.opt.res_layers,
                norm=self.opt.norm_mlp,
                last_op=nn.Sigmoid() if not self.cfg.test_mode else None,
                args=args
            )
        else:
            self.if_regressor = MLP(
                filter_channels=channels_IF,
                name="if",
                res_layers=self.opt.res_layers,
                norm=self.opt.norm_mlp,
                last_op=nn.Sigmoid() if not self.cfg.test_mode else None,
                args=args
            )

        self.sp_encoder = SpatialEncoder()

        # network
        if self.use_filter:
            if self.opt.gtype == "HGPIFuNet":
                self.F_filter = HGFilter(self.opt, self.opt.num_stack, len(self.channels_filter[0]))
            else:
                print(colored(f"Backbone {self.opt.gtype} is unimplemented", "green"))

        summary_log = (
            f"{self.prior_type.upper()}:\n" + f"w/ Global Image Encoder: {self.use_filter}\n" +
            f"Image Features used by MLP: {self.in_geo}\n"
        )

        if self.prior_type == "icon":
            summary_log += f"Geometry Features used by MLP: {self.smpl_feats}\n"
            summary_log += f"Dim of Image Features (local): {3 if (use_vis and not self.use_filter) else 6}\n"
            summary_log += f"Dim of Geometry Features (ICON): {self.smpl_dim}\n"
        elif self.prior_type == "keypoint":
            summary_log += f"Geometry Features used by MLP: {self.smpl_feats}\n"
            summary_log += f"Dim of Image Features (local): {3 if (use_vis and not self.use_filter) else 6}\n"
            summary_log += f"Dim of Geometry Features (Keypoint): {self.smpl_dim}\n"
        elif self.prior_type == "pamir":
            summary_log += f"Dim of Image Features (global): {self.hourglass_dim}\n"
            summary_log += f"Dim of Geometry Features (PaMIR): {self.voxel_dim}\n"
        else:
            summary_log += f"Dim of Image Features (global): {self.hourglass_dim}\n"
            summary_log += f"Dim of Geometry Features (PIFu): 1 (z-value)\n"

        summary_log += f"Dim of MLP's first layer: {channels_IF[0]}\n"

        print(colored(summary_log, "yellow"))

        self.normal_filter = NormalNet(self.cfg)
        n_bins=8
        self.mark = math.log(6)/math.log(10)
        self.bins = torch.logspace(0,self.mark, n_bins)-1
        self.bins=self.bins.cuda()

        init_net(self)

    def get_normal(self, in_tensor_dict):

        # insert normal features
        # if (not self.training) and (not self.overfit):
                        
            with torch.no_grad():
                feat_lst = []
                if "image" in self.in_geo:
                    feat_lst.append(in_tensor_dict["image"])    # [1, 3, 512, 512]

                # if (
                #         "normal_F" not in in_tensor_dict.keys() and
                #         "normal_B" not in in_tensor_dict.keys()
                #     ):
                #     print("img only")
                # elif (
                #         "normal_F" not in in_tensor_dict.keys() or
                #         "normal_B" not in in_tensor_dict.keys()
                #     ):
                #         (nmlF, nmlB) = self.normal_filter(in_tensor_dict)
                #         if nmlF!=None: 
                #             feat_lst.append(nmlF)    # [1, 3, 512, 512]
                #         if nmlB!=None: 
                #             feat_lst.append(nmlB)    # [1, 3, 512, 512]
                # else:
                #             print("icon training using ground truth normal??")
                            
                #             nmlF = in_tensor_dict["normal_F"]
                #             feat_lst.append(nmlF)
                #             nmlB = in_tensor_dict["normal_B"]
                #             feat_lst.append(nmlB) 
                if self.prior_type in ["icon", "keypoint"]:
                    if (
                            "normal_F"  in in_tensor_dict.keys() and
                            "normal_B"  in in_tensor_dict.keys()
                        ):      
                        nmlF = in_tensor_dict["normal_F"]# [1, 3, 512, 512]
                        feat_lst.append(nmlF)
                        nmlB = in_tensor_dict["normal_B"]# [1, 3, 512, 512]
                        feat_lst.append(nmlB) 
    
                    elif (
                            "normal_F" not in in_tensor_dict.keys() or
                            "normal_B" not in in_tensor_dict.keys()
                        ):
                            (nmlF, nmlB) = self.normal_filter(in_tensor_dict)
                            if nmlF!=None: 
                                feat_lst.append(nmlF)    # [1, 3, 512, 512]
                            if nmlB!=None: 
                                feat_lst.append(nmlB)    # [1, 3, 512, 512]
                
                            
            in_filter = torch.cat(feat_lst, dim=1)

        # else:
        #     in_filter = torch.cat([in_tensor_dict[key] for key in self.in_geo], dim=1)

            return in_filter

    def get_mask(self, in_filter, size=128):

        mask = (
            F.interpolate(
                in_filter[:, self.channels_filter[0]],
                size=(size, size),
                mode="bilinear",
                align_corners=True,
            ).abs().sum(dim=1, keepdim=True) != 0.0
        )

        return mask

    def filter(self, in_tensor_dict, return_inter=False):
        """
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        """

        in_filter = self.get_normal(in_tensor_dict) #the cmap is from the ground truth smpl model.

        features_G = []

        if self.prior_type in ["icon", "keypoint"]:
            if self.use_filter:
                features_F = self.F_filter(
                    in_filter[:, self.channels_filter[0]]
                )    # [(B,hg_dim,128,128) * 4]
                features_B = self.F_filter(
                    in_filter[:, self.channels_filter[1]]
                )    # [(B,hg_dim,128,128) * 4]
            else:
                features_F = [in_filter[:, self.channels_filter[0]]]
                features_B = [in_filter[:, self.channels_filter[1]]]
            for idx in range(len(features_F)):
                features_G.append(torch.cat([features_F[idx], features_B[idx]], dim=1))
        else:
            if self.use_filter:
                    features_G = self.F_filter(in_filter[:, self.channels_filter[0]])
            else:
                features_G = [in_filter[:, self.channels_filter[0]]]

        self.smpl_feat_dict = {
            k: in_tensor_dict[k] if k in in_tensor_dict.keys() else None
            for k in getattr(self, f"{self.prior_type}_keys")
        }

        # If it is not in training, only produce the last im_feat
        if not self.training:
            features_out = [features_G[-1]]
        else:
            features_out = features_G

        if return_inter:
            return features_out, in_filter
        else:
            return features_out

    def query(self, features, points, calibs, transforms=None, regressor=None, clip_feature=None):
        xyz = self.projection(points, calibs, transforms)
        (xy, z) = xyz.split([2, 1], dim=1)

        in_cube = (xyz > -1.0) & (xyz < 1.0)
        in_cube = in_cube.all(dim=1, keepdim=True).detach().float()

        preds_list = []
        vol_feats = features

        if self.prior_type in ["icon", "keypoint"]: ##icon is slow due to the Point feat following code

            # smpl_verts [B, N_vert, 3]
            # smpl_faces [B, N_face, 3]
            # xyz [B, 3, N]  --> points [B, N, 3]

            point_feat_extractor = PointFeat(
                self.smpl_feat_dict["smpl_verts"], self.smpl_feat_dict["smpl_faces"], args=self.args
            )

            point_feat_out = point_feat_extractor.query(
                xyz.permute(0, 2, 1).contiguous(), self.smpl_feat_dict
            )
            
            feat_lst = [
                point_feat_out[key] for key in self.smpl_feats if key in point_feat_out.keys()
            ]
            smpl_feat = torch.cat(feat_lst, dim=2).permute(0, 2, 1)

            if self.prior_type == "keypoint":
                kpt_feat = self.sp_encoder.forward(
                    cxyz=xyz.permute(0, 2, 1).contiguous(),
                    kptxyz=self.smpl_feat_dict["smpl_joint"],
                )

        elif self.prior_type == "pamir":

            voxel_verts = self.smpl_feat_dict["voxel_verts"][:, :-self.
                                                             smpl_feat_dict["pad_v_num"][0], :]
            voxel_faces = self.smpl_feat_dict["voxel_faces"][:, :-self.
                                                             smpl_feat_dict["pad_f_num"][0], :]

            self.voxelization.update_param(
                batch_size=voxel_faces.shape[0],
                smpl_tetra=voxel_faces[0].detach().cpu().numpy(),
            )
            vol = self.voxelization(voxel_verts)    # vol ~ [0,1]
            vol_feats = self.ve(vol, intermediate_output=self.training)

        for im_feat, vol_feat in zip(features, vol_feats):
            # normal feature choice by smpl_vis

            if self.prior_type == "icon":
                if "vis" in self.smpl_feats:
                    
                    point_local_feat = feat_select(self.index(im_feat, xy), smpl_feat[:, [-1], :])
                    point_feat_list = [point_local_feat, smpl_feat[:, :-1, :]]
                else:
                    point_local_feat = self.index(im_feat, xy)
                    point_feat_list = [point_local_feat, smpl_feat[:, :, :]]

            if self.prior_type == "keypoint":

                if "vis" in self.smpl_feats:
                    point_local_feat = feat_select(self.index(im_feat, xy), smpl_feat[:, [-1], :])
                    point_feat_list = [point_local_feat, kpt_feat, smpl_feat[:, :-1, :]]
                else:
                    point_local_feat = self.index(im_feat, xy)
                    point_feat_list = [point_local_feat, kpt_feat, smpl_feat[:, :, :]]

            elif self.prior_type == "pamir":

                # im_feat [B, hg_dim, 128, 128]
                # vol_feat [B, vol_dim, 32, 32, 32]

                point_feat_list = [self.index(im_feat, xy), self.index(vol_feat, xyz)]

            elif self.prior_type == "pifu":
                point_feat_list = [self.index(im_feat, xy), z]

            point_feat = torch.cat(point_feat_list, 1)
            # out of image plane is always set to 0
            if self.args.use_clip:
                   preds = regressor(point_feat, clip_feature) 
            else: preds = regressor(point_feat) ###sdf feature in channle [:,6,:]
            preds = in_cube * preds

            preds_list.append(preds)

        return preds_list

    # def get_error(self, preds_if_list, labels):
    #     """calcaulate error

    #     Args:
    #         preds_list (list): list of torch.tensor(B, 3, N)
    #         labels (torch.tensor): (B, N_knn, N)

    #     Returns:
    #         torch.tensor: error
    #     """
    #     error_if = 0

    #     for pred_id in range(len(preds_if_list)):
    #         pred_if = preds_if_list[pred_id]
    #         error_if += self.error_term(pred_if, labels)

    #     error_if /= len(preds_if_list)

    #     return error_if

    def get_error(self, preds_if_list, labels):
        """calcaulate error

        Args:
            preds_list (list): list of torch.tensor(B, 3, N)
            labels (torch.tensor): (B, N_knn, N)

        Returns:
            torch.tensor: error
        """
        error_if = 0
        # error_if_uncertain = 0
        for pred_id in range(len(preds_if_list)):
            pred_if = preds_if_list[pred_id]
            if pred_if.size(1)==1:
                error_if += self.error_term(pred_if, labels)
            elif pred_if.size(1)==2:
                # beta=pred_if[:,1:2,:] + self.args.beta_min
                # error_if +=((pred_if[:,:1,:]-labels)**2/(2*beta**2)).mean()
                # error_if += self.args.beta_plus + torch.log(beta).mean() # +3 to make it positive
                beta=pred_if[:,1:2,:]
                beta_exp=torch.exp(beta)
                # error_if +=((pred_if[:,:1,:]-labels)**2/(beta_exp)).mean()
                l1_loss=F.mse_loss(pred_if[:,:1,:],labels,reduce=False)
                error_if +=(l1_loss/beta_exp).mean() 
                error_if += beta.mean() #+3 #to make it positive
                # print(beta.mean(),"beta.mean()", beta_exp, 'beta_exp')  #tensor(-3.4107, device='cuda:0') beta.mean() tensor([[[0.0745, 0.0627, 0.0730,  ..., 0.0225, 0.0444, 0.0225]]],          device='cuda:0') beta_exp                                                                                                     | 5/66 [02:01<15:59, 15.73s/it]

        error_if /= len(preds_if_list)

        if self.args.kl_div and pred_if.size(1)==2:
                disp_loss = l1_loss.flatten()
                uncert_loss = beta_exp.flatten() # act: relu, uncert_est is log(sigma)

                disp_p, uncert_p = soft_assignment(disp_loss, uncert_loss, self.bins)
                kl_loss = F.kl_div(uncert_p.log(),disp_p, reduction='sum')
                return error_if+kl_loss, kl_loss

        return error_if

    def forward(self, in_tensor_dict):
        """
        sample_tensor [B, 3, N]
        calib_tensor [B, 4, 4]
        label_tensor [B, 1, N]
        smpl_feat_tensor [B, 59, N]
        """

        sample_tensor = in_tensor_dict["sample"]
        calib_tensor = in_tensor_dict["calib"]
        label_tensor = in_tensor_dict["label"]

        in_feat = self.filter(in_tensor_dict)
        ####
        if self.args.use_clip:
            clip_feature=in_tensor_dict["clip_feature"]
            preds_if_list = self.query(
            in_feat, sample_tensor, calib_tensor, regressor=self.if_regressor, clip_feature=clip_feature
        )
        ####
        else: preds_if_list = self.query(
            in_feat, sample_tensor, calib_tensor, regressor=self.if_regressor
        )
        error = self.get_error(preds_if_list, label_tensor)
        return preds_if_list[-1], error

def sample_from_planes(plane_axes, plane_features, coordinates, mode='bilinear', padding_mode='zeros', box_warp=None):
        assert padding_mode == 'zeros'
        N, n_planes, C, H, W = plane_features.shape
        _, M, _ = coordinates.shape #bs, num_points, xyz_cord
        plane_features = plane_features.view(N*n_planes, C, H, W) #bs*n_planes, channel, height, width

        # coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds

        projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1) #bs*n_plane, none, num_points, uv cordinate on each plane
        output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)#bs, num_planes, num_points, channels
        return output_features
def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes, _, _ = planes.shape
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N*n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N*n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    return projections[..., :2]
if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    # plane_features=torch.randn(8,3,4,128,128).cuda()
    # coordinates=torch.rand(8,8000,3).cuda()
    # plane_axes=torch.tensor([[[1, 0, 0],
    #                         [0, 1, 0],
    #                         [0, 0, 1]],
    #                         [[1, 0, 0],
    #                         [0, 0, 1],
    #                         [0, 1, 0]],
    #                         [[0, 0, 1],
    #                         [1, 0, 0],
    #                         [0, 1, 0]]], dtype=torch.float32).cuda()
    # a=sample_from_planes(plane_axes,plane_features,coordinates,box_warp=1)#1,3,8000,4
    # print(1)
    plane_features=torch.randn(8,3,32,32,32).cuda()
    coordinates=torch.rand(8,8000,3).cuda()
    plane_axes=torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]]], dtype=torch.float32).cuda()
    a=sample_from_planes(plane_axes,plane_features,coordinates,box_warp=1)#1,3,8000,4
    print(1)

