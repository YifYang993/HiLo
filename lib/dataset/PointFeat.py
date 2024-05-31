from pytorch3d.structures import Meshes
import torch.nn.functional as F
import torch
from lib.common.render_utils import face_vertices
from lib.dataset.mesh_util import SMPLX, barycentric_coordinates_of_projection
from kaolin.ops.mesh import check_sign
from kaolin.metrics.trianglemesh import point_to_mesh_distance
import numpy as np
import time

smplx=SMPLX()
import torch.nn as nn

class PosEmbedding(nn.Module):
    def __init__(self, max_logscale, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super().__init__()
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freqs = 2**torch.linspace(0, max_logscale, N_freqs)
        else:
            self.freqs = torch.linspace(1, 2**max_logscale, N_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (B, 3)

        Outputs:
            out: (B, 6*N_freqs+3)
        """
        out = [x]
        for freq in self.freqs:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)
    
class PosEmbedding(nn.Module):
    def __init__(self, max_logscale, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super().__init__()
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freqs = 2**torch.linspace(0, max_logscale, N_freqs)
        else:
            self.freqs = torch.linspace(1, 2**max_logscale, N_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (B, 3)

        Outputs:
            out: (B, 6*N_freqs+3)
        """
        out = [x]
        for freq in self.freqs:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)
    

class adaptive_positional_encoding(nn.Module):
        def __init__(self, L, barf_c2f=[0.1, 0.5]) -> None:
            super().__init__()
            self.L=L
            self.barf_c2f=barf_c2f
            print(self.barf_c2f)
            self.progress = torch.nn.Parameter(torch.tensor(0.))

        def forward(self, input): # [B,...,N]  ##use this
            shape = input.shape
            freq = 2**torch.arange(self.L, dtype=torch.float32, device=input.device)*np.pi # [L]
            spectrum = input[...,None]*freq # [B,...,N,L]
            sin,cos = spectrum.sin(),spectrum.cos() # [B,...,N,L]
            input_enc = torch.stack([sin,cos],dim=-2) # [B,...,N,2,L]
            input_enc = input_enc.view(*shape[:-1],-1) # [B,...,2NL]
            # coarse-to-fine: smoothly mask positional encoding for BARF
                # set weights for different frequency bands
            start,end = self.barf_c2f
            alpha = (self.progress.data-start)/(end-start)*self.L
            # print("self.progress.data",self.progress.data)
            k = torch.arange(self.L,dtype=torch.float32, device=input.device)
            weight = (1-(alpha-k).clamp_(min=0,max=1).mul_(np.pi).cos_())/2
            # apply weights
            shape = input_enc.shape
            input_enc = (input_enc.view(-1,self.L)*weight).view(*shape)
            input_enc=torch.cat([input,input_enc], dim=-1)
            return input_enc
        
class PointFeat:
    def __init__(self, verts, faces, args=None, adaptive_positional_encoding=None):

        # verts [B, N_vert, 3]
        # faces [B, N_face, 3]
        # triangles [B, N_face, 3, 3]
        if args:
            self.args=args
        self.Bsize = verts.shape[0]
        self.mesh = Meshes(verts, faces)
        self.device = verts.device
        self.faces = faces
        # SMPL has watertight mesh, but SMPL-X has two eyeballs and open mouth
        # 1. remove eye_ball faces from SMPL-X: 9928-9383, 10474-9929
        # 2. fill mouth holes with 30 more faces
        
        if verts.shape[1] == 10475:
            faces = faces[:, ~smplx.smplx_eyeball_fid]  ##modify this to save time, all eye id in smplx are the same
            # faces1 = faces[:, ~SMPLX().smplx_eyeball_fid]
            mouth_faces = (
                torch.as_tensor(smplx.smplx_mouth_fid, device=self.device).unsqueeze(0).repeat(self.Bsize, 1,
                                                                             1)
            )
            self.faces = torch.cat([faces, mouth_faces], dim=1).long()

        self.verts = verts
        self.triangles = face_vertices(self.verts, self.faces)
        # breakpoint()
        if self.args.adaptive_pe_sdf:
            self.embedding_sdf = adaptive_positional_encoding
        else:
            self.embedding_sdf = PosEmbedding(self.args.PE_sdf-1, self.args.PE_sdf)
        # self.embedding_sdf = adaptive_positional_encoding#(L=self.args.PE_sdf)
        # breakpoint()

    def query(self, points, feats={}):
        """
        Given the predicted clothed-body normal maps, ̂ N c, and the SMPL-body mesh, M, we regress the implicit 3D surface of a clothed human based on local features FP: FP = [Fs(P), F b n (P), F c n (P)], (6) where Fs is the signed distance from a query point P to the closest body point Pb ∈ M, and F b n is the barycentric surface normal of Pb; both provide strong regularization against self occlusions. Finally, Fc n is a normal vector extracted from ̂ Nc front or ̂ Nc back depending on the visibility of Pb: Fc n (P) = { ̂ Nc front(π(P)) if Pb is visible ̂ Nc back(π(P)) else, (7) where π(P) denotes the 2D projection of the 3D point P.
        """
        # points [B, N, 3]
        # feats {'feat_name': [B, N, C]}
        del_keys = ["smpl_verts", "smpl_faces", "smpl_joint", "voxel_verts", "voxel_faces","pad_v_num", "pad_f_num"]

        residues, pts_ind, _ = point_to_mesh_distance(points, self.triangles)
        closest_triangles = torch.gather(
            self.triangles, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, 3)
        ).view(-1, 3, 3)
        bary_weights, barypoints = barycentric_coordinates_of_projection(points.view(-1, 3), closest_triangles)

        out_dict = {}

        for feat_key in feats.keys():

            if feat_key in del_keys:
                continue

            elif feats[feat_key] is not None:
                feat_arr = feats[feat_key]
                feat_dim = feat_arr.shape[-1]
                feat_tri = face_vertices(feat_arr, self.faces)
                closest_feats = torch.gather(
                    feat_tri, 1, pts_ind[:, :, None, None].expand(-1, -1, 3, feat_dim)
                ).view(-1, 3, feat_dim)
                pts_feats = ((closest_feats * bary_weights[:, :, None]).sum(1).unsqueeze(0))  ##try to remove the weights
                out_dict[feat_key.split("_")[1]] = pts_feats

            else:
                out_dict[feat_key.split("_")[1]] = None

        if "sdf" in out_dict.keys(): ###add noise to it 
            pts_dist = torch.sqrt(residues) / torch.sqrt(torch.tensor(3))
            pts_signs = 2.0 * (check_sign(self.verts, self.faces[0], points).float() - 0.5)
            pts_sdf = (pts_dist * pts_signs).unsqueeze(-1)
            if self.args.perturb_sdf!=0:
                perturb=(torch.rand(pts_sdf.size(), device=pts_sdf.device) -0.5) *2 *self.args.perturb_sdf
                pts_sdf+=perturb
            if self.args.PE_sdf!=0:
                pts_sdf= self.embedding_sdf(pts_sdf)
            # breakpoint()
            if self.args.sdfdir:
                assert points.size(0)*points.size(1)==barypoints.size(0)
                sdf_direction = points - barypoints.view(points.size(0), points.size(1), 3)
                # out_dict["sdfdir"] = sdf_direction
                pts_sdf=torch.cat([pts_sdf, sdf_direction], dim=-1)
            
            out_dict["sdf"] = pts_sdf

        if "vis" in out_dict.keys():
            out_dict["vis"] = out_dict["vis"].ge(1e-1).float()

        if "norm" in out_dict.keys():
            temp_=torch.tensor([-1.0, 1.0, -1.0],device=self.device)
            pts_norm = out_dict["norm"] * temp_
            out_dict["norm"] = F.normalize(pts_norm, dim=2)

        if "cmap" in out_dict.keys():
            out_dict["cmap"] = out_dict["cmap"].clamp_(min=0.0, max=1.0)

        for out_key in out_dict.keys():
            out_dict[out_key] = out_dict[out_key].view(self.Bsize, -1, out_dict[out_key].shape[-1])

        return out_dict
    

if __name__ == "__main__":
    APE=adaptive_positional_encoding(L=6)
    APE=PosEmbedding(5,6)
    input=torch.rand(2, 100, 1)
    # output=APE.positional_encoding(input)
    output=APE(input)
    print(output.shape)

