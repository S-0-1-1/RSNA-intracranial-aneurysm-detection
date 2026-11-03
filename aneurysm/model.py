import json
import math
import os

import torch
import torch.nn as nn

from aneurysm.labels import MODALITIES, N_TERR

RESENC_L = dict(
    n_stages=6, features_per_stage=[32, 64, 128, 256, 320, 320], kernel_sizes=[[3, 3, 3]] * 6,
    strides=[[1, 1, 1]] + [[2, 2, 2]] * 5, n_blocks_per_stage=[1, 3, 4, 6, 6, 6],
    n_conv_per_stage_decoder=[1] * 5, conv_bias=True,
)


def build_resenc_l():
    from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
    return ResidualEncoderUNet(
        input_channels=1, num_classes=1, conv_op=nn.Conv3d, norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={'eps': 1e-5, 'affine': True}, dropout_op=None, dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU, nonlin_kwargs={'inplace': True}, deep_supervision=False, **RESENC_L)


def load_pretrained_encoder(path):
    ckp = torch.load(path, map_location='cpu', weights_only=True)
    weights = ckp.get('network_weights')
    if weights is None and any(str(k).startswith('encoder.') for k in ckp):
        weights = ckp
    if weights is None:
        raise KeyError(f'{path}: no network_weights entry')
    plan = ckp.get('nnssl_adaptation_plan')
    side = os.path.join(os.path.dirname(path), 'adaptation_plan.json')
    if plan is None and os.path.exists(side):
        with open(side) as f:
            plan = json.load(f)
    if plan is None:
        raise KeyError(f'{path}: no nnssl_adaptation_plan entry and no adaptation_plan.json next to it')
    arch = plan['architecture_plans']['arch_class_name']
    if arch not in ('ResEncL', 'ResidualEncoderUNet'):
        raise ValueError(f'{path}: architecture {arch} is not ResEnc-L')
    if int(plan['pretrain_num_input_channels']) != 1:
        raise ValueError(f'{path}: pretrained with {plan["pretrain_num_input_channels"]} input channels, expected 1')
    sd = {}
    for k, v in weights.items():
        for prefix in ('module.', '_orig_mod.'):
            if k.startswith(prefix):
                k = k[len(prefix):]
        sd[k] = v
    net = build_resenc_l()
    net.load_state_dict(sd, strict=True)
    return net.encoder


class CBAM3d(nn.Module):
    def __init__(self, c, reduction=8, kernel=7):
        super().__init__()
        h = max(c // reduction, 8)
        self.mlp = nn.Sequential(nn.Conv3d(c, h, 1), nn.LeakyReLU(inplace=True), nn.Conv3d(h, c, 1))
        self.spatial = nn.Conv3d(2, 1, kernel, padding=kernel // 2)
        self.gate = nn.Parameter(torch.zeros(1, c, 1, 1, 1))

    def forward(self, x):
        ca = torch.sigmoid(self.mlp(x.mean((2, 3, 4), keepdim=True)) + self.mlp(x.amax((2, 3, 4), keepdim=True)))
        y = x * ca
        sa = torch.sigmoid(self.spatial(torch.cat([y.mean(1, keepdim=True), y.amax(1, keepdim=True)], 1)))
        return x + self.gate * (y * sa - x)


def conv_norm_act(cin, cout, k=3):
    return nn.Sequential(nn.Conv3d(cin, cout, k, padding=k // 2), nn.InstanceNorm3d(cout, eps=1e-5, affine=True),
                         nn.LeakyReLU(inplace=True))


class UpStage(nn.Module):
    def __init__(self, cin, cskip):
        super().__init__()
        self.up = nn.ConvTranspose3d(cin, cskip, 2, 2)
        self.att = CBAM3d(cskip)
        self.block = conv_norm_act(2 * cskip, cskip)

    def forward(self, x, skip):
        return self.block(torch.cat([self.up(x), self.att(skip)], 1))


class PositionPrior(nn.Module):
    def __init__(self, out_ch, freqs=(1, 2, 4, 8), hidden=64):
        super().__init__()
        self.register_buffer('freqs', torch.tensor(freqs, dtype=torch.float32) * math.pi, persistent=False)
        cin = 3 + 6 * len(freqs)
        self.net = nn.Sequential(nn.Conv3d(cin, hidden, 1), nn.LeakyReLU(inplace=True),
                                 nn.Conv3d(hidden, hidden, 1), nn.LeakyReLU(inplace=True), nn.Conv3d(hidden, out_ch, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, c):
        b, _, d, h, w = c.shape
        f = c[:, :, None] * self.freqs.view(1, 1, -1, 1, 1, 1)
        f = f.reshape(b, -1, d, h, w)
        return self.net(torch.cat([c, torch.sin(f), torch.cos(f)], 1))


class Head(nn.Module):
    def __init__(self, cin, cout, bias=0.0):
        super().__init__()
        self.body = conv_norm_act(cin, cin // 2)
        self.out = nn.Conv3d(cin // 2, cout, 1)
        nn.init.constant_(self.out.bias, bias)

    def forward(self, x):
        return self.out(self.body(x))


class AneurysmNet(nn.Module):
    def __init__(self, encoder, prior_logit=-6.9):
        super().__init__()
        self.encoder = encoder
        f = list(encoder.output_channels)
        self.ups = nn.ModuleList([UpStage(f[s], f[s - 1]) for s in range(len(f) - 1, 1, -1)])
        self.hm_head = Head(f[1], 1, prior_logit)
        self.terr_head = Head(f[1], N_TERR)
        self.aux_head = nn.Conv3d(f[2], 1, 1)
        nn.init.constant_(self.aux_head.bias, prior_logit)
        self.prior = PositionPrior(1 + N_TERR)
        self.mod_head = nn.Linear(2 * f[-1], len(MODALITIES))

    def forward(self, x, coords):
        skips = self.encoder(x)
        y = skips[-1]
        outs = []
        for i, up in enumerate(self.ups):
            y = up(y, skips[-2 - i])
            outs.append(y)
        prior = self.prior(coords.float())
        hm = self.hm_head(y) + prior[:, :1]
        terr = self.terr_head(y) + prior[:, 1:]
        g = skips[-1]
        mod = self.mod_head(torch.cat([g.mean((2, 3, 4)), g.amax((2, 3, 4))], 1))
        return dict(hm=hm, terr=terr, aux=self.aux_head(outs[-2]), mod=mod)


def build_model(pretrained_path, cfg):
    encoder = load_pretrained_encoder(pretrained_path) if pretrained_path else build_resenc_l().encoder
    return AneurysmNet(encoder, cfg['prior_logit'])


def param_groups(model, weight_decay):
    enc, new = [], []
    for name, p in model.named_parameters():
        (enc if name.startswith('encoder.') else new).append(p)
    return [dict(params=enc, weight_decay=weight_decay, name='encoder'), dict(params=new, weight_decay=weight_decay, name='new')]


def readout(hm, terr, k, mask=None):
    p = torch.sigmoid(hm.float()).flatten(1)
    t = torch.softmax(terr.float()[:, 1:], 1).flatten(2)
    if mask is not None:
        p = p * mask.flatten(1).float()
    q = p[:, None] * t
    k = min(k, p.shape[1])
    return p.topk(k, 1).values.mean(1), q.topk(k, 2).values.mean(2)
