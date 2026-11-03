EMPTY_FRAC = 0.005

PREPROCESSING = dict(
    size=256,
    margin_mm=4.0,
    cube_min_mm=140.0,
    cube_max_mm=210.0,
    keep_mm=20.0,
    dist_unit_mm=0.5,
    ct_window=(-100.0, 900.0),
    ct_clip=(-1024.0, 3071.0),
    mr_pct=(0.5, 99.95),
    overview_mm=4.0,
    head_mm=220.0,
    head_xy_mm=150.0,
    body_margin_mm=15.0,
    no_brain='roi',
    min_brain_cm3=30.0,
    min_slices=5,
    strict=True,
    fov_mm=220.0,
    gap_break=3.0,
    ps_table=None,
)

TRAINING = dict(
    cube=256,
    patch=128,
    keep_mm=20.0,
    dist_unit_mm=0.5,
    pos_frac=0.5,
    loc_power=0.5,
    mod_power=0.5,
    hard_neg=0.75,
    neg_sigma=14.0,
    neg_box=56.0,
    pos_jitter=40.0,
    edge=24.0,
    border=12.0,
    p_flip=0.5,
    p_rot=0.85,
    rot_deg=15.0,
    p_scale=0.85,
    scale=(0.85, 1.15),
    strip_mm=(6.0, 20.0),
    strip_inf_mm=12.0,
    p_bias_mr=0.3,
    p_bias_ct=0.1,
    bias_max=0.35,
    p_bright=0.3,
    bright=(0.85, 1.15),
    p_contrast=0.3,
    contrast=(0.75, 1.25),
    p_gamma=0.3,
    gamma=(0.7, 1.5),
    p_lowres=0.25,
    lowres=(1.5, 4.0),
    lowres_max_gap=1.5,
    p_noise=0.15,
    noise=(0.0, 0.03),
    hm_sigma_mm=3.0,
    ball_mm=4.0,
    terr_dilate=2,
    readout_k=8,
    prior_logit=-6.9,
    w_hm=1.0,
    w_aux=0.3,
    w_terr=0.5,
    w_ap=0.1,
    w_loc=0.1,
    w_mod=0.05,
    focal_gamma=2.0,
    hm_norm_floor=50.0,
)


def merge(defaults, overrides):
    unknown = set(overrides) - set(defaults)
    if unknown:
        raise KeyError(f'unknown config keys {sorted(unknown)}')
    return {**defaults, **overrides}


def prep_cfg(**overrides):
    return merge(PREPROCESSING, overrides)


def train_cfg(**overrides):
    return merge(TRAINING, overrides)
