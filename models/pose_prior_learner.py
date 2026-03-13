import sys
sys.path.append("..")
sys.path.append("../..")

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.optim import Adam
import torchvision.models as tvmodels
from models.modules import ResNetConditionalParameterRegressor, ResNetReconstructor
from utils.losses import compute_template_boundary_loss, VGGPerceptualLoss
from utils.utils import  draw_lines
import math

class MLPBlock(nn.Module):
    def __init__(self, dim, inter_dim, dropout_ratio):
        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(dim, inter_dim),
            nn.GELU(),
            nn.Dropout(dropout_ratio),
            nn.Linear(inter_dim, dim),
            nn.Dropout(dropout_ratio)
        )

    def forward(self, x):
        return self.ff(x)

class MixerLayer(nn.Module):
    def __init__(self,
                 hidden_dim,
                 hidden_inter_dim,
                 token_dim,
                 token_inter_dim,
                 dropout_ratio=0.0):
        super().__init__()

        self.layernorm1 = nn.LayerNorm(hidden_dim)
        self.MLP_token = MLPBlock(token_dim, token_inter_dim, dropout_ratio)
        self.layernorm2 = nn.LayerNorm(hidden_dim)
        self.MLP_channel = MLPBlock(hidden_dim, hidden_inter_dim, dropout_ratio)

    def forward(self, x):
        y = self.layernorm1(x)
        y = y.transpose(2, 1)
        y = self.MLP_token(y)
        y = y.transpose(2, 1)
        z = self.layernorm2(x + y)
        z = self.MLP_channel(z)
        out = x + y + z
        return out

class Memory(nn.Module):
    def __init__(self, num_parts):
        super(Memory, self).__init__()
        self.decay = 0.9
        self.num_parts = num_parts
        self.token_num = 34
        self.token_class_num_each = 16
        self.token_dim = 512

        self.enc_hidden_dim = 512
        self.enc_hidden_inter_dim = 512
        self.enc_token_inter_dim = 64
        self.enc_num_blocks = 4
        self.encoder_start = nn.Linear(2, self.enc_hidden_dim)
        self.encoder = nn.ModuleList(
            [MixerLayer(self.enc_hidden_dim, self.enc_hidden_inter_dim,
                        self.num_parts, self.enc_token_inter_dim
                        ) for _ in range(self.enc_num_blocks)])
        self.encoder_layer_norm = nn.LayerNorm(self.enc_hidden_dim)
        self.token_mlp = nn.Linear(self.num_parts, self.token_num)
        self.feature_embed = nn.Linear(self.enc_hidden_dim, self.token_dim)

        self.dec_hidden_dim = 32
        self.dec_hidden_inter_dim = 64
        self.dec_token_inter_dim = 64
        self.dec_num_blocks = 1
        self.decoder_token_mlp = nn.Linear(self.token_num, self.num_parts)
        self.decoder_start = nn.Linear(self.token_dim, self.dec_hidden_dim)
        self.decoder = nn.ModuleList(
            [MixerLayer(self.dec_hidden_dim, self.dec_hidden_inter_dim,
                        self.num_parts, self.dec_token_inter_dim
                        ) for _ in range(self.dec_num_blocks)])
        self.decoder_layer_norm = nn.LayerNorm(self.dec_hidden_dim)
        self.recover_embed = nn.Linear(self.dec_hidden_dim, 2)

        self.register_buffer('codebook', torch.empty(self.token_num, self.token_class_num_each, self.token_dim)) #(k, 16, d)
        self.codebook.data.normal_()
        self.register_buffer('ema_cluster_size', torch.zeros(self.token_num, self.token_class_num_each)) #(k, 16)
        self.register_buffer('ema_w', torch.empty(self.token_num, self.token_class_num_each, self.token_dim)) #(k, 16, d)
        self.ema_w.data.normal_()

    def kpt2token(self, points):
        encode_feat = self.encoder_start(points)
        for num_layer in self.encoder:
            encode_feat = num_layer(encode_feat)
        encode_feat = self.encoder_layer_norm(encode_feat)

        encode_feat = encode_feat.transpose(2, 1)
        encode_feat = self.token_mlp(encode_feat).transpose(2, 1)
        encode_feat = self.feature_embed(encode_feat)
        return encode_feat

    def token2kpt(self, token_feat):

        token_feat = token_feat.transpose(2, 1)
        token_feat = self.decoder_token_mlp(token_feat).transpose(2, 1)
        decode_feat = self.decoder_start(token_feat)

        for num_layer in self.decoder:
            decode_feat = num_layer(decode_feat)
        decode_feat = self.decoder_layer_norm(decode_feat)

        recoverd_points = self.recover_embed(decode_feat)
        return recoverd_points

    def forward(self, x):
        # x(b, n, 2) n=16, k=34 codebook(k, 64, d)
        encode_feat = self.kpt2token(x) # ze (b, k, d)
        encode_feat = encode_feat.transpose(1, 0) # ze (k, b, d)
        distances = torch.sum(encode_feat ** 2, dim=2, keepdim=True) \
                    + torch.sum(self.codebook.transpose(2, 1) ** 2, dim=1, keepdim=True) \
                    - 2 * torch.matmul(encode_feat, self.codebook.transpose(2, 1))  # (k, b, 64)

        encoding_indices = torch.argmin(distances, dim=2)  # (k, b)
        encodings = torch.zeros(encoding_indices.shape[0], encoding_indices.shape[1], self.token_class_num_each).cuda()
        encodings.scatter_(2, encoding_indices.unsqueeze(2), 1)  # (k, b, 64)
        token_feat = torch.matmul(encodings, self.codebook)

        dw = torch.matmul(encodings.transpose(2, 1), encode_feat.detach()) #(k, 64, b) x (k, b, d) = (k, 64, d)
        self.ema_cluster_size = self.ema_cluster_size * self.decay + (1 - self.decay) * torch.sum(encodings, 1) #(k, 64)
        n = torch.sum(self.ema_cluster_size.data, dim=-1, keepdim=True) #(k, 1)
        self.ema_cluster_size = ((self.ema_cluster_size + 1e-5) / (n + self.token_class_num_each * 1e-5) * n)
        self.ema_w = self.ema_w * self.decay + (1 - self.decay) * dw #(k, 64, d)
        self.codebook = self.ema_w / self.ema_cluster_size.unsqueeze(-1)

        commitment_loss = F.mse_loss(token_feat.detach(), encode_feat)
        vq_loss = commitment_loss
        token_feat = encode_feat + (token_feat - encode_feat).detach()
        token_feat = token_feat.transpose(1, 0)  # zq (b, k, d)
        output = self.token2kpt(token_feat)  # (b, n, 2)
        return output, vq_loss

    def get_template(self):
        template_token_feat = self.codebook.detach().mean(dim=1, keepdim=True) #(k, 1, d)
        template_token_feat = template_token_feat.transpose(1, 0) #(1, k, d)
        output = self.token2kpt(template_token_feat) #(1, n, 2)
        return output.squeeze().detach() #(n, 2)


class PosePriorLearner(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_parts = args.num_parts
        self.thick = args.thick
        self.sklr = args.sklr
        self.skeleton_idx = torch.triu_indices(self.num_parts, self.num_parts, offset=1)
        self.n_skeleton = len(self.skeleton_idx[0])
        skeleton_scalar = (torch.randn(self.num_parts, self.num_parts) - 4) / self.sklr
        self.skeleton_scalar = nn.Parameter(skeleton_scalar)

        self.memory = Memory(num_parts=self.num_parts)
        self.aug = torch.ones(self.num_parts, 1).cuda()
        self.regressor = ResNetConditionalParameterRegressor(num_parts=args.num_parts)
        self.translator = ResNetReconstructor()
        self.vgg_loss = VGGPerceptualLoss()
        #self.I = torch.eye(3)[0:2].view(1, 1, 2, 3).repeat(args.batch_size, args.num_parts, 1, 1).cuda()

        self.block = args.block
        self.missing = args.missing
        self.use_alpha = args.use_alpha
        if args.use_alpha:
            self.alpha = nn.Parameter(torch.tensor(args.alpha), requires_grad=True)

    def skeleton_scalar_matrix(self) -> torch.Tensor:
        """
        Give the skeleton scalar matrix
        :return: (num_parts, num_parts)
        """
        skeleton_scalar = F.softplus(self.skeleton_scalar * self.sklr)
        skeleton_scalar = torch.triu(skeleton_scalar, diagonal=1)
        skeleton_scalar = skeleton_scalar + skeleton_scalar.transpose(1, 0)
        return skeleton_scalar

    def rasterize(self, keypoints: torch.Tensor, output_size: int = 128) -> torch.Tensor:
        """
        Generate edge heatmap from keypoints, where edges are weighted by the learned scalars.
        :param keypoints: (batch_size, n_points, 2)
        :return: (batch_size, 1, heatmap_size, heatmap_size)
        """

        paired_joints = torch.stack([keypoints[:, self.skeleton_idx[0], :2], keypoints[:, self.skeleton_idx[1], :2]],
                                    dim=2)
        skeleton_scalar = F.softplus(self.skeleton_scalar * self.sklr)
        skeleton_scalar = torch.triu(skeleton_scalar, diagonal=1)
        skeleton_scalar = skeleton_scalar[self.skeleton_idx[0], self.skeleton_idx[1]].reshape(1, self.n_skeleton, 1, 1)

        skeleton_heatmap_sep = draw_lines(paired_joints, heatmap_size=output_size, thick=self.thick)
        skeleton_heatmap_sep = skeleton_heatmap_sep * skeleton_scalar.reshape(1, self.n_skeleton, 1, 1)
        skeleton_heatmap = skeleton_heatmap_sep.max(dim=1, keepdim=True)[0]
        return skeleton_heatmap

    def normalize_points(self, points):
        # p(b, n, 2)
        n = points.size(1)
        px_max = torch.max(points[:, :, 0], dim=-1, keepdim=True).values
        px_min = torch.min(points[:, :, 0], dim=-1, keepdim=True).values
        py_max = torch.max(points[:, :, 1], dim=-1, keepdim=True).values
        py_min = torch.min(points[:, :, 1], dim=-1, keepdim=True).values
        px = (points[:, :, 0] - px_min) / (px_max - px_min)
        py = (points[:, :, 1] - py_min) / (py_max - py_min)
        p = torch.stack([px, py], dim=-1)
        return p

    def compute_edge_reg_loss(self, t_points, points):
        t_points = self.normalize_points(t_points)
        points = self.normalize_points(points)
        len_t = torch.sqrt(((t_points[:, self.skeleton_idx[0], :2] - t_points[:, self.skeleton_idx[1], :2]) ** 2).sum(dim=-1))
        len = torch.sqrt(((points[:, self.skeleton_idx[0], :2] - points[:, self.skeleton_idx[1], :2]) ** 2).sum(dim=-1))
        valid_edge = (len < 0.7) * (len > 0.3)
        len_diff = torch.abs(len_t - len).detach()
        skeleton_scalar = F.softplus(self.skeleton_scalar * self.sklr)
        skeleton_scalar = torch.triu(skeleton_scalar, diagonal=1)
        skeleton_scalar = skeleton_scalar[self.skeleton_idx[0], self.skeleton_idx[1]].reshape(1, self.n_skeleton)
        loss = torch.abs((10 * len_diff + torch.log(skeleton_scalar)) * valid_edge).mean()
        return loss

    def forward(self, frame, return_imgs=False):

        # Get batch_size, number of joints and image dimension
        batch_size = frame.shape[0]
        img_size = frame.shape[2]

        # Estimate the regressor parameters
        template_points = self.memory.get_template().unsqueeze(0)
        estimated_params = self.regressor(frame, template_points.repeat(batch_size, 1, 1))
        #estimated_params = self.I + estimated_params

        aug = self.aug.unsqueeze(0).repeat(batch_size, 1, 1)
        transformed_template_points = torch.matmul(estimated_params,
                                                   torch.cat([template_points.repeat(batch_size, 1, 1), aug], dim=-1).unsqueeze(-1)).squeeze(-1)
        boundary_loss = compute_template_boundary_loss(transformed_template_points)
        #edge_reg_loss = self.compute_edge_reg_loss(template_points.detach(), transformed_template_points.detach())
        transformed_template = self.rasterize(transformed_template_points, output_size=img_size)

        damage_mask = torch.zeros(frame.shape[0], 1, self.block, self.block,
                                  device=frame.device).uniform_() > self.missing
        damage_mask = F.interpolate(damage_mask.to(frame), size=frame.shape[-1], mode='nearest')
        ref_frame = frame * damage_mask
        if self.use_alpha:
            ref_frame = ref_frame * self.alpha
        reconstructed = self.translator(ref_frame, transformed_template)

        vq_transformed_template_points, vq_loss = self.memory(transformed_template_points.detach())
        consistency =  100 * F.mse_loss(vq_transformed_template_points, transformed_template_points.detach())

        transformed_template = transformed_template.repeat(1, 3, 1, 1)
        template = self.rasterize(template_points, output_size=img_size).repeat(1, 3, 1, 1)
        vq_transformed_template = self.rasterize(vq_transformed_template_points, output_size=img_size)
        vq_transformed_template = vq_transformed_template.repeat(1, 3, 1, 1)

        recon_perceptual_loss = self.vgg_loss(reconstructed, frame)
        loss = recon_perceptual_loss + \
               boundary_loss + \
               consistency

        d = {'boundary_loss': boundary_loss,
             'recon_perceptual_loss': recon_perceptual_loss,
             #'edge_reg_loss': edge_reg_loss,
             'consistency': consistency}
        loss = loss + vq_loss
        d['vq_loss'] = vq_loss
        if return_imgs:
            d['template'] = template.detach()
            d['reconstructed'] = reconstructed
            d['frame'] = frame
            d['ref_frame'] = ref_frame
            d['transformed_template'] = transformed_template.detach()
            d['vq_transformed_template'] = vq_transformed_template.detach()

        return loss, d