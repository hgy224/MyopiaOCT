import collections

import torch
import torch.nn as nn
from torch.nn import Parameter

import models_vit
import torch.nn.functional as F


class MyopiaOCTNet(nn.Module):
    def __init__(self, vit_model_name='vit_large_patch16', in_chans=12, classes_num=4, drop_path_rate=0.1,
                 conv_out_channels=(10, 10), conv_ksizes=(2, 3), scaling_factors=(0.75, 1.0, 1.25),
                 reduction_ratio=16,
                 top_k=5):
        super(MyopiaOCTNet, self).__init__()
        self.scaling_factors = scaling_factors
        # ViT
        self.feature_extractor = models_vit.__dict__[vit_model_name](
                                        in_chans=in_chans,
                                        drop_path_rate=drop_path_rate,
                                    )
        primary_feature_channels = self.feature_extractor.num_features
        self.primary_feature_h = self.feature_extractor.img_size // self.feature_extractor.patch_size


        # multi-scale conv
        conv2ds = []
        batch_norms = []
        self.secondary_feature_channels = 0
        for i, ks in enumerate(conv_ksizes):
            conv_out_channel = conv_out_channels[i]
            conv2ds.append(nn.Conv2d(primary_feature_channels, conv_out_channel, kernel_size=ks, padding=0))
            for j in range(len(scaling_factors)):
                batch_norms.append(nn.BatchNorm2d(conv_out_channel, affine=False,
                                                  track_running_stats=True, momentum=0.05))
                self.secondary_feature_channels += conv_out_channel

        self.conv2ds = nn.ModuleList(conv2ds)
        self.batch_norms = nn.ModuleList(batch_norms)
        self.scalar_multipliers = Parameter(torch.FloatTensor(len(batch_norms)), requires_grad=True)
        self.scalar_multipliers.data.fill_(1.0)
        self.scalar_multipliers.data = self.scalar_multipliers.data * len(self.scalar_multipliers) / self.scalar_multipliers.sum().item()

        self.attention_fusion = AttentionFusion(self.secondary_feature_channels, reduction_ratio)

        self.avg_topk_pool = AvgTopKPool(self.secondary_feature_channels, top_k)

        self.fc = nn.Linear(self.secondary_feature_channels, classes_num)

    def forward(self, x):
        primary_feature = self.feature_extractor(x)
        ph = self.primary_feature_h
        sec_features = []
        for j, scaling_factor in enumerate(self.scaling_factors):
            hp = int(scaling_factor * ph)
            resized_feature = nn.functional.interpolate(primary_feature, (hp, hp), mode='bilinear', align_corners=True)
            for i in range(len(self.conv2ds)):
                sec_fea_i = self.conv2ds[i](resized_feature)
                batch_norm = self.batch_norms[i*len(self.scaling_factors)+j]
                sec_fea_i_bn = batch_norm(sec_fea_i)
                scalar_multiplier = self.scalar_multipliers[i*len(self.scaling_factors)+j]
                sec_fea_i_bn_sm = scalar_multiplier * sec_fea_i_bn
                sec_fea = nn.functional.interpolate(sec_fea_i_bn_sm, (ph, ph), mode='bilinear', align_corners=True)
                sec_features.append(sec_fea)
        sec_features = torch.cat(sec_features, dim=1)

        thr_features = self.attention_fusion(sec_features)

        feature_vector = self.avg_topk_pool(thr_features)

        return self.fc(feature_vector)


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_types=('avg', 'max')):
        super(ChannelAttention, self).__init__()
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            else:
                raise ValueError('pool type should be avg or max')

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        kernel_size = 7
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1,
                                      padding=(kernel_size - 1) // 2)

        self.bn = nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        x_out = self.spatial_conv(x_compress)
        x_out = self.relu(self.bn(x_out))
        scale = F.sigmoid(x_out)
        return x * scale


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class AttentionFusion(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, pool_types=('avg', 'max')):
        super(AttentionFusion, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio, pool_types)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.channel_attention(x)
        return self.spatial_attention(x)


class AvgTopKPool(nn.Module):
    def __init__(self, channels, top_k, decay=0.6):
        super(AvgTopKPool, self).__init__()
        self.top_k = top_k
        weight = []
        for c in range(channels):
            for k in range(top_k):
                weight.append(decay ** k)
        weights = torch.tensor(weight, requires_grad=True).view(channels, top_k)
        self.weights = Parameter(weights / torch.sum(weights, dim=1).view(channels, 1), requires_grad=True)

    def forward(self, x):
        x_flat = x.view(x.shape[0], x.shape[1], -1)
        x_topk = torch.topk(x_flat, self.top_k, dim=2)[0]
        x_avgtopk = (x_topk * self.weights).sum(dim=2)
        return x_avgtopk


if __name__ == '__main__':
    # a = torch.rand((32, 12, 224, 224))
    model = MyopiaOCTNet()
    checkpoint = torch.load('/data/home/huanggengyou/mae/output_large/checkpoint-399.pth', map_location='cpu')['model']

    checkpoint = collections.OrderedDict([('feature_extractor.' + k, v) if k.startswith('blocks') or k.startswith('cls_token') or k.startswith('pos_embed') or k.startswith('patch_embed') else (k, v) for k, v in checkpoint.items()])
    # print(checkpoint.keys())
    # print(model.state_dict().keys())
    msg = model.load_state_dict(checkpoint, strict=False)
    print(msg)
