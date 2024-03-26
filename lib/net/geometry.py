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

import torch


def index(feat, uv):
    '''
    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [0, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    uv = uv.transpose(1, 2)    # [B, N, 2]

    (B, N, _) = uv.shape
    C = feat.shape[1]

    if uv.shape[-1] == 3:
        # uv = uv[:,:,[2,1,0]]
        # uv = uv * torch.tensor([1.0,-1.0,1.0]).type_as(uv)[None,None,...]
        uv = uv.unsqueeze(2).unsqueeze(3)    # [B, N, 1, 1, 3]
    else:
        uv = uv.unsqueeze(2)    # [B, N, 1, 2]

    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)    # [B, C, N, 1]
    return samples.view(B, C, N)    # [B, C, N] 1,12,8000   max 0.0693  min -0.0544



def sample_from_planes(plane_axes, plane_features, coordinates, vis, mode='bilinear', padding_mode='zeros', box_warp=1):
        assert padding_mode == 'zeros'
        N, n_planes, C, H = plane_features.shape[0],plane_features.shape[1],plane_features.shape[2],plane_features.shape[3:]
        _, M, _ = coordinates.shape #bs, num_points, xyz_cord
        if vis==True:
            plane_features_f=plane_features[:,:,:C//2,...]
            plane_features_b=plane_features[:,:,C//2:,...]
            if len(H)==3:
                plane_features_f = plane_features_f.view(N*n_planes, C//2, H[0], H[1], H[2]) #bs*n_planes, channel, height, width
                plane_features_b = plane_features_b.view(N*n_planes, C//2, H[0], H[1], H[2]) #bs*n_planes, channel, height, width
            else:
                plane_features_f = plane_features_f.view(N*n_planes, C//2,  H[0], H[1]) 
                plane_features_b = plane_features_b.view(N*n_planes, C//2,  H[0], H[1]) 
            # coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds
            projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1) 
            plane_features_f = torch.nn.functional.grid_sample(plane_features_f, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False)#.permute(0, 3, 2, 1).reshape(N, n_planes, M, C)#bs, num_planes, num_points, channels
            plane_features_b = torch.nn.functional.grid_sample(plane_features_b, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False)#.permute(0, 3, 2, 1).reshape(N, n_planes, M, C)#bs, num_planes, num_points, channels
            plane_features_f=plane_features_f.view(N, n_planes, C//2, M).mean(1)
            plane_features_b=plane_features_b.view(N, n_planes, C//2, M).mean(1)
            return torch.cat([plane_features_f,plane_features_b], dim=1)

        else:
            if len(H)==3:
                plane_features = plane_features.view(N*n_planes, C, H[0], H[1], H[2]) #bs*n_planes, channel, height, width
            else:
                plane_features = plane_features.view(N*n_planes, C,  H[0], H[1]) #bs*n_planes, channel, height, width

            coordinates = (2/box_warp) * coordinates # TODO: add specific box bounds

            projected_coordinates = project_onto_planes(plane_axes, coordinates).unsqueeze(1) #bs*n_plane, none, num_points, uv cordinate on each plane
            output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates.float(), mode=mode, padding_mode=padding_mode, align_corners=False)#.permute(0, 3, 2, 1).reshape(N, n_planes, M, C)#bs, num_planes, num_points, channels
            output_features=output_features.view(N, n_planes, C, M).mean(1)
            # output_features=output_features.transpose(0, 1) #3.8557  max()   -1.1015 min() view(n_planes*C, N, M).
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

def index_triplane(feat, xyz, vis=True): 
    '''
    :param feat: [B, C, H, W] image features
    :param xyz: [B, 3, N] xyz coordinates in the image plane, range [0, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    (B, N, _) = xyz.shape
    B, C, H = feat.shape[0], feat.shape[1], list(feat.shape[2:])
    xyz = xyz.transpose(1, 2) 
    plane_axes=torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [1, 0, 0],
                            [0, 1, 0]]], dtype=torch.float32, device=feat.device)
    if len(H)==2:
        feat=feat.view(B, 3, C//3, H[0],H[1])
    else:
        feat=feat.view(B, 3, C//3, H[0], H[1], H[2])
    samples=sample_from_planes(plane_axes, feat, xyz, vis=vis)#1,3,8000,4
    return samples   #3.8557  max()   -1.1015 min()


    # uv = uv.transpose(1, 2)    # [B, N, 2]

    # (B, N, _) = uv.shape
    # C = feat.shape[1]

    # if uv.shape[-1] == 3:
    #     # uv = uv[:,:,[2,1,0]]
    #     # uv = uv * torch.tensor([1.0,-1.0,1.0]).type_as(uv)[None,None,...]
    #     uv = uv.unsqueeze(2).unsqueeze(3)    # [B, N, 1, 1, 3]
    # else:
    #     uv = uv.unsqueeze(2)    # [B, N, 1, 2]

    # # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # # for old versions, simply remove the aligned_corners argument.
    # samples = torch.nn.functional.grid_sample(feat, uv, align_corners=True)    # [B, C, N, 1]
    # return samples.view(B, C, N)    # [B, C, N] 



def orthogonal(points, calibrations, transforms=None):
    '''
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 3, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)    # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts


def perspective(points, calibrations, transforms=None):
    '''
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [Bx3x4] Tensor of projection matrix
    :param transforms: [Bx2x3] Tensor of image transform matrix
    :return: xy: [Bx2xN] Tensor of xy coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    homo = torch.baddbmm(trans, rot, points)    # [B, 3, N]
    xy = homo[:, :2, :] / homo[:, 2:3, :]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        xy = torch.baddbmm(shift, scale, xy)

    xyz = torch.cat([xy, homo[:, 2:3, :]], 1)
    return xyz
