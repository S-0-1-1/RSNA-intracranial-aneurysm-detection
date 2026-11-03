import torch
import torch.nn.functional as F

from aneurysm.labels import N_TERR
from aneurysm.model import readout


def focal_bce(logits, target, gamma):
    p = torch.sigmoid(logits)
    return (target - p).abs().pow(gamma) * F.binary_cross_entropy_with_logits(logits, target, reduction='none')


def prob_bce(p, y, w):
    p = p.clamp(1e-6, 1 - 1e-6)
    loss = -(y * torch.log(p) + (1 - y) * torch.log1p(-p))
    return (loss * w).sum() / w.sum().clamp(min=1.0)


def compute_losses(out, tgt, cfg):
    hm = out['hm'].float()
    l_hm = focal_bce(hm, tgt['hm'], cfg['focal_gamma']).sum() / tgt['hm'].sum().clamp(min=cfg['hm_norm_floor'])
    aux = out['aux'].float()
    l_aux = focal_bce(aux, tgt['aux'], cfg['focal_gamma']).sum() / tgt['aux'].sum().clamp(min=cfg['hm_norm_floor'] / 8)
    terr = out['terr'].float()
    ce = F.cross_entropy(terr, tgt['terr'], reduction='none')
    lab = tgt['terr'].flatten()
    wf = tgt['terr_w'].flatten()
    cls_sum = torch.zeros(N_TERR, device=ce.device).index_add(0, lab, ce.flatten() * wf)
    cls_cnt = torch.zeros(N_TERR, device=ce.device).index_add(0, lab, wf)
    present = (cls_cnt > 0).float()
    l_terr = ((cls_sum / cls_cnt.clamp(min=1.0)) * present).sum() / present.sum().clamp(min=1.0)
    p_ap, p_loc = readout(hm, terr, cfg['readout_k'])
    l_ap = prob_bce(p_ap, tgt['y_ap'], tgt['w_ap'])
    l_loc = prob_bce(p_loc, tgt['y_loc'], tgt['w_loc'])
    l_mod = F.cross_entropy(out['mod'].float(), tgt['mod'])
    return (cfg['w_hm'] * l_hm + cfg['w_aux'] * l_aux + cfg['w_terr'] * l_terr + cfg['w_ap'] * l_ap
            + cfg['w_loc'] * l_loc + cfg['w_mod'] * l_mod)
