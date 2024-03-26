# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn
import pytorch_lightning as pl
from termcolor import colored
import torch.nn.functional as F
import math


class MLP3d(pl.LightningModule):
    def __init__(self, filter_channels, name=None, res_layers=[], norm='group', last_op=None, args=None):

        super(MLP3d, self).__init__()
        self.args=args
        if args.mlp_first_dim!=0:
            filter_channels[0]=args.mlp_first_dim
            print(colored("I have modified mlp filter channles{}".format(filter_channels),"red"))
        self.filters = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_layers = res_layers
        self.norm = norm
        self.last_op = last_op
        self.name = name
        self.activate = nn.LeakyReLU(inplace=True)
###############se module
        assert [self.args.mlpSe, self.args.mlpSev1, self.args.mlpSemax].count(True) in [0,1], "mlp se strategy cannot be embodied simultaneously"
        if self.args.mlpSe: ##this strategy yields best results, while not surpasses baseline yet. 
            self.se_conv = nn.ModuleList()
            for filters_nums_ in filter_channels[:-3]:
                self.se_conv.append(SpatialSELayer(filters_nums_))  #1449 gpu memory for bs 2
            for filters_nums_ in filter_channels[-3:]:
                self.se_conv.append(SpatialSELayer3d(filters_nums_))  #1449 gpu memory for bs 2
                # self.se_conv.append(ChannelSELayer(filters_nums_))  #1457 gpu memory for bs 2
        elif self.args.mlpSev1:
            self.se_conv = nn.ModuleList()
            for filters_nums_ in filter_channels[:-1]:
                # self.se_conv.append(SpatialSELayer(filters_nums_))  #1449 gpu memory for bs 2
                self.se_conv.append(ChannelSELayer(filters_nums_))  #1457 gpu memory for bs 2
        elif self.args.mlpSemax:
            self.se_conv_spatial = nn.ModuleList()
            self.se_conv_channel = nn.ModuleList()
            for filters_nums_ in filter_channels[:-1]:
                self.se_conv_spatial.append(SpatialSELayer(filters_nums_))  #1449 gpu memory for bs 2
                self.se_conv_channel.append(ChannelSELayer(filters_nums_)) 
################
        for l in range(0, len(filter_channels) - 3):
            if l in self.res_layers:
                self.filters.append(
                    nn.Conv1d(filter_channels[l] + filter_channels[0], filter_channels[l + 1], 1)
                )
            else:
                self.filters.append(nn.Conv1d(filter_channels[l], filter_channels[l + 1], 1))

            if l != len(filter_channels) - 2:
                if norm == 'group':
                    self.norms.append(nn.GroupNorm(32, filter_channels[l + 1]))
                elif norm == 'batch':
                    self.norms.append(nn.BatchNorm1d(filter_channels[l + 1]))
                elif norm == 'instance':
                    self.norms.append(nn.InstanceNorm1d(filter_channels[l + 1]))
                elif norm == 'weight':
                    self.filters[l] = nn.utils.weight_norm(self.filters[l], name='weight')
                    # print(self.filters[l].weight_g.size(),
                    #       self.filters[l].weight_v.size())
        for l in range(len(filter_channels) - 3, len(filter_channels) - 1):
            if l in self.res_layers:
                self.filters.append(
                    CNN3D(filter_channels[l] + filter_channels[0], filter_channels[l + 1], 1)
                )
            else:
                self.filters.append(CNN3D(filter_channels[l], filter_channels[l + 1], 1))

            if l != len(filter_channels) - 2:
                if norm == 'group':
                    self.norms.append(nn.GroupNorm(32, filter_channels[l + 1]))
                elif norm == 'batch':
                    self.norms.append(nn.BatchNorm3d(filter_channels[l + 1]))
                elif norm == 'instance':
                    self.norms.append(nn.InstanceNorm3d(filter_channels[l + 1]))
                elif norm == 'weight':
                    self.filters[l] = nn.utils.weight_norm(self.filters[l], name='weight')
        self.len_filter=len(self.filters)

    def forward(self, feature):
        '''
        feature may include multiple view inputs
        args:
            feature: [B, C_in, N]
        return:
            [B, C_out, N] prediction
        '''

        y = feature
        tmpy = feature
        bs,c,num_p=tmpy.size()
        cuberoot=int(round(math.pow(num_p, 1.0/3.0)))
        tmpy=tmpy.view(bs,c,cuberoot, cuberoot, cuberoot)
        for i, f in enumerate(self.filters):
            if self.args.mlpSe or self.args.mlpSev1:
                if i!=self.len_filter-1:
                    y=self.se_conv[i](y) 
            elif self.args.mlpSemax:
                if i!=self.len_filter-1:
                    y_spa=self.se_conv_spatial[i](y) ##
                    y_cha=self.se_conv_channel[i](y) ##
                    y=torch.max(y_spa, y_cha)
            y = f(y if i not in self.res_layers else torch.cat([y, tmpy], 1))
            if i != self.len_filter - 1:
                if self.norm not in ['batch', 'group', 'instance']:
                    y = self.activate(y)
                else:
                    y = self.activate(self.norms[i](y))
            if i ==self.len_filter - 3:
                  bs,c,num_p=y.size()
                  y=y.view(bs,c,cuberoot, cuberoot, cuberoot)
   
        if self.last_op is not None:
            y = self.last_op(y)
        y=y.view(bs,1,-1)

        return y
    
import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, dim),
        #     nn.Dropout(dropout)
        # )

        self.l1=nn.Linear(dim, hidden_dim)
        self.g1=nn.GELU()
        self.d1=nn.Dropout(dropout)
        self.l2=nn.Linear(hidden_dim, dim)
        self.d2=nn.Dropout(dropout)

    def forward(self, x):
        # return self.net(x)
        x=self.l1(x)
        x=self.g1(x)
        x=self.d1(x)
        x=self.l2(x)
        x=self.d2(x)
        return x

class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
        super().__init__()

        # self.token_mix = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     Rearrange('b n d -> b d n'),
        #     FeedForward(num_patch, token_dim, dropout),
        #     Rearrange('b d n -> b n d')
        # )

        # self.channel_mix = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     FeedForward(dim, channel_dim, dropout),
        # )

        self.ln1=nn.LayerNorm(dim)
        self.r1=Rearrange('b n d -> b d n')
        self.f1=FeedForward(num_patch, token_dim, dropout)
        self.r2=Rearrange('b d n -> b n d')



        self.ln2=nn.LayerNorm(dim)
        self.f2=FeedForward(dim, channel_dim, dropout)

    def forward(self, x):
        tempx=x
        x=self.ln1(x)
        x=self.r1(x)
        x=self.f1(x)
        x=self.r2(x)
        x = tempx + x
        tempx1=x
        x=self.ln2(x)
        x=self.f2(x)
        x = tempx1 + x

        return x


class MLPMixer(nn.Module):   #def __init__(self, filter_channels, name=None, res_layers=[], norm='group', last_op=None, args=None):

    def __init__(self, filter_channels, name=None, num_classes=1, depth=8, token_dim=256, channel_dim=2048, res_layers=[], norm='group', last_op=None, args=None): 
        super().__init__()

        self.num_patch =  8000
        # self.to_patch_embedding = nn.Sequential(
        #     nn.Conv2d(in_channels, dim, patch_size, patch_size),
        #     Rearrange('b c h w -> b (h w) c'),
        # )
        filter_channels0=filter_channels[0]
        self.mixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self.mixer_blocks.append(MixerBlock(filter_channels0, self.num_patch, token_dim, channel_dim))

        self.layer_norm = nn.LayerNorm(filter_channels0)

        self.mlp_head = nn.Sequential(
            nn.Linear(filter_channels0, num_classes)
        )

    def forward(self, x):  ##x size (1,13,8000)    bs channel points  <->  bs channel patch


        # x = self.to_patch_embedding(x)
        x=Rearrange('b n d -> b d n')(x) #put channel dimension behind point dimension
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        x=self.mlp_head(x)
        x=Rearrange('b n d -> b d n')(x) 
        # x = self.layer_norm(x)
        return x
        # x = x.mean(dim=1)

        # return self.mlp_head(x)





    

class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv1d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, *a= input_tensor.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, *a)
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor

class SpatialSELayer3d(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer3d, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, *a= input_tensor.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv3d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, *a)
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor



class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        # self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2) ###cannot mix features of different points toghther

        # channel excitation
        # fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_1 = self.fc1(squeeze_tensor)
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1))
        return output_tensor

if __name__=="__main__":
    # class args_():
    #     def __init__(self) -> None:
    #         self.test_code=True
    #         self.mlp_first_dim=12
    #         self.mlpSev1=False
    #         self.mlpSe=True
    #         self.mlpSemax=False
    # args_=args_()
    # net=MLP3d(filter_channels=[12, 512, 256, 128, 1], res_layers= [2,3,4],args=args_).cuda()
    # # net=MLP(filter_channels=[12,128,256,128,1], args=args_).cuda()
    # input=torch.randn(2,12,8000).cuda()
    # print(net(input).size())
    # print(1)

    # if __name__ == "__main__":
    img = torch.ones([1, 13, 340031])
    tokensize=img.size(-1)
    dim_=[img.size(-2)]
    results=[]
    model = MLPMixer(num_classes=1,
                     filter_channels=dim_, depth=8, token_dim=256, channel_dim=2048)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3fM' % parameters)
    bs, c, B=img.size()
    chunk_temp=8000
    for i in range(0, B, chunk_temp):
        img_chunk=img[:,:,i:i+chunk_temp]
        print(i,img_chunk.size(-1))
        if img_chunk.size(-1)!=chunk_temp:
            img_chunk=torch.cat([img_chunk, torch.ones(bs,c,chunk_temp-img_chunk.size(-1))],dim=2)
        out_img_chunk = model(img_chunk)

        results.append(out_img_chunk)
    out_img=torch.cat(results,2)[:,:,:B]
    print("Shape of out :", out_img.shape)  # [B, in_channels, image_size, image_size]