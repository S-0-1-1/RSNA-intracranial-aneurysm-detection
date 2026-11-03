import json
from pathlib import Path

import numpy as np
import pydicom
from pydicom.pixels.decoders import RLELosslessDecoder

pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE
if {'pydicom', 'pylibjpeg'} <= set(RLELosslessDecoder.available_plugins):
    RLELosslessDecoder.remove_plugin('pydicom')

TEST_TAGS = frozenset([
    'BitsAllocated', 'BitsStored', 'Columns', 'FrameOfReferenceUID', 'HighBit', 'ImageOrientationPatient',
    'ImagePositionPatient', 'InstanceNumber', 'Modality', 'PatientID', 'PerFrameFunctionalGroupsSequence',
    'PhotometricInterpretation', 'PixelRepresentation', 'PixelSpacing', 'PlanarConfiguration',
    'RescaleIntercept', 'RescaleSlope', 'RescaleType', 'Rows', 'SOPClassUID', 'SOPInstanceUID',
    'SamplesPerPixel', 'SliceThickness', 'SpacingBetweenSlices', 'StudyInstanceUID', 'TransferSyntaxUID',
])
DECODE_TAGS = frozenset(['PixelData', 'NumberOfFrames'])

CANONICAL_AXES = {
    'axial': (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    'coronal': (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])),
    'sagittal': (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0])),
}


def unit(v):
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v)
    if n < 1e-8:
        raise ValueError('zero-length direction vector')
    return v / n


def read_dicom(path, strict, pixels):
    ds = pydicom.dcmread(path, stop_before_pixels=not pixels)
    if strict:
        keep = TEST_TAGS | DECODE_TAGS
        for tag in [e.tag for e in ds if e.keyword not in keep]:
            del ds[tag]
    return ds


def first_item(seq):
    if seq is None or len(seq) == 0:
        return None
    return seq[0]


def lookup(ds, i, seq_name, attr):
    pf = ds.get('PerFrameFunctionalGroupsSequence')
    if pf is not None and i < len(pf):
        item = first_item(pf[i].get(seq_name))
        if item is not None and item.get(attr) is not None:
            return item.get(attr), 'perframe'
    sf = ds.get('SharedFunctionalGroupsSequence')
    if sf is not None and len(sf) > 0:
        item = first_item(sf[0].get(seq_name))
        if item is not None and item.get(attr) is not None:
            return item.get(attr), 'shared'
    if ds.get(attr) is not None:
        return ds.get(attr), 'top'
    return None, 'missing'


def as_vector(x):
    if x is None:
        return None
    return np.array([float(t) for t in x], np.float64)


def as_float(x, default):
    if x is None or x == '':
        return default
    return float(x)


def frame_count(ds):
    if ds.get('NumberOfFrames') is not None and str(ds.NumberOfFrames) != '':
        return max(1, int(ds.NumberOfFrames))
    pf = ds.get('PerFrameFunctionalGroupsSequence')
    if pf is not None and len(pf) > 0:
        return len(pf)
    return 1


def frames_of(path, ds):
    n = frame_count(ds)
    multi = n > 1 or ds.get('PerFrameFunctionalGroupsSequence') is not None
    sop = str(ds.get('SOPInstanceUID', ''))
    inst = ds.get('InstanceNumber')
    inst = int(inst) if inst is not None and str(inst) != '' else 0
    out = []
    for i in range(n):
        ipp, s_ipp = lookup(ds, i, 'PlanePositionSequence', 'ImagePositionPatient')
        iop, s_iop = lookup(ds, i, 'PlaneOrientationSequence', 'ImageOrientationPatient')
        ps, s_ps = lookup(ds, i, 'PixelMeasuresSequence', 'PixelSpacing')
        th, _ = lookup(ds, i, 'PixelMeasuresSequence', 'SliceThickness')
        sb, _ = lookup(ds, i, 'PixelMeasuresSequence', 'SpacingBetweenSlices')
        sl, _ = lookup(ds, i, 'PixelValueTransformationSequence', 'RescaleSlope')
        ic, _ = lookup(ds, i, 'PixelValueTransformationSequence', 'RescaleIntercept')
        out.append(dict(
            path=path, index=i, multi=multi, sop=sop, inst=inst,
            rows=int(ds.Rows), cols=int(ds.Columns),
            ipp=as_vector(ipp), iop=as_vector(iop), ps=as_vector(ps),
            thick=as_float(th, None), sbs=as_float(sb, None),
            slope=as_float(sl, 1.0), inter=as_float(ic, 0.0),
            src_ipp=s_ipp, src_iop=s_iop, src_ps=s_ps,
        ))
    return out


def position_step(frames):
    pos = np.array([f['ipp'] for f in frames if f['ipp'] is not None])
    if len(pos) < 2:
        return None
    steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    steps = steps[steps > 1e-3]
    return round(float(np.median(steps)), 2) if steps.size else None


def table_key(rows, cols, step):
    return f'{rows}x{cols}@{step}'


def table_spacing(frames, cfg):
    table = cfg.get('ps_table')
    if not table:
        return None, None
    rows, cols = frames[0]['rows'], frames[0]['cols']
    entry = table.get(table_key(rows, cols, position_step(frames)))
    if entry is not None:
        return np.array(entry['ps'], np.float64), 'table'
    same = [v for k, v in table.items() if k.startswith(f'{rows}x{cols}@')]
    if same:
        best = max(same, key=lambda v: v['count'])
        return np.array(best['ps'], np.float64), 'table_shape'
    return None, None


def dominant_plane(normal):
    return ('sagittal', 'coronal', 'axial')[int(np.argmax(np.abs(normal)))]


def complete_geometry(frames, cfg):
    src = {}
    have_ipp = [f['ipp'] is not None for f in frames]
    have_iop = [f['iop'] is not None for f in frames]
    have_ps = [f['ps'] is not None for f in frames]
    if all(have_iop):
        src['iop'] = frames[0]['src_iop']
    elif any(have_iop):
        ref = next(f['iop'] for f in frames if f['iop'] is not None)
        for f in frames:
            if f['iop'] is None:
                f['iop'] = ref.copy()
        src['iop'] = 'partial'
    else:
        pos = np.array([f['ipp'] for f in frames if f['ipp'] is not None])
        if len(pos) >= 2 and np.ptp(pos, axis=0).max() > 1e-3:
            _, _, vt = np.linalg.svd(pos - pos.mean(0), full_matrices=False)
            normal = unit(vt[0])
            u0, v0 = CANONICAL_AXES[dominant_plane(normal)]
            u = unit(u0 - np.dot(u0, normal) * normal)
            v = unit(np.cross(normal, u))
            if np.dot(v, v0) < 0:
                v = -v
            iop = np.concatenate([u, v])
            src['iop'] = 'derived'
        else:
            iop = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            src['iop'] = 'assumed'
        for f in frames:
            f['iop'] = iop.copy()
    if all(have_ps):
        src['ps'] = frames[0]['src_ps']
    else:
        value, how = table_spacing(frames, cfg)
        for f in frames:
            if f['ps'] is None:
                f['ps'] = value.copy() if value is not None else np.full(2, cfg['fov_mm'] / max(f['rows'], f['cols']))
        src['ps'] = (how if value is not None else 'assumed') if not any(have_ps) else 'partial'
    for f in frames:
        f['u'] = unit(f['iop'][:3])
        v = f['iop'][3:] - np.dot(f['iop'][3:], f['u']) * f['u']
        f['v'] = unit(v)
    if all(have_ipp):
        src['ipp'] = frames[0]['src_ipp']
    elif any(have_ipp):
        raise ValueError('ImagePositionPatient present for some frames only')
    else:
        step = frames[0]['sbs'] or frames[0]['thick'] or 1.0
        order = sorted(range(len(frames)), key=lambda j: (frames[j]['path'], frames[j]['index']))
        for rank, j in enumerate(order):
            n = np.cross(frames[j]['u'], frames[j]['v'])
            frames[j]['ipp'] = rank * step * n
        src['ipp'] = 'assumed'
    return src


def same_stack(a, b):
    return (a['rows'] == b['rows'] and a['cols'] == b['cols']
            and np.abs(a['ps'] - b['ps']).max() < 0.01
            and np.dot(a['u'], b['u']) > 0.999 and np.dot(a['v'], b['v']) > 0.999)


def split_stacks(frames):
    stacks = []
    for f in frames:
        for s in stacks:
            if same_stack(s[0], f):
                s.append(f)
                break
        else:
            stacks.append([f])
    return sorted(stacks, key=len, reverse=True)


def list_series_files(series_dir):
    paths = [p for p in Path(series_dir).rglob('*') if p.is_file() and p.suffix.lower() in ('.dcm', '')]
    if not paths:
        raise FileNotFoundError(f'no DICOM files in {series_dir}')
    return sorted(str(p) for p in paths)


def read_headers(paths, cfg):
    frames = []
    first = None
    for p in paths:
        ds = read_dicom(p, cfg['strict'], False)
        if first is None:
            first = ds
        frames.extend(frames_of(p, ds))
    return frames, first


def read_pixels(path, cfg):
    ds = read_dicom(path, cfg['strict'], True)
    nf = frame_count(ds)
    if nf > 1 and ds.get('NumberOfFrames') is None:
        ds.NumberOfFrames = nf
    return ds.pixel_array, nf


def read_full(path, cfg):
    ds = read_dicom(path, cfg['strict'], True)
    frames = frames_of(path, ds)
    nf = frame_count(ds)
    if nf > 1 and ds.get('NumberOfFrames') is None:
        ds.NumberOfFrames = nf
    return frames, ds.pixel_array, nf


def frame_count_of(frames, sop):
    return sum(1 for f in frames if f['sop'] == sop)


def build_plan(series_dir, cfg, points=None, headers=None):
    paths = list_series_files(series_dir)
    frames, first = read_headers(paths, cfg) if headers is None else headers
    src = complete_geometry(frames, cfg)
    stacks = split_stacks(frames)
    stack = stacks[0]
    u, v = stack[0]['u'], stack[0]['v']
    n = np.cross(u, v)
    d_all = np.array([np.dot(f['ipp'], n) for f in stack])
    order = np.lexsort((np.array([f['index'] for f in stack]), np.array([f['inst'] for f in stack]), d_all))
    kept, last = [], None
    for j in order:
        if last is not None and abs(d_all[j] - last) < 0.05:
            continue
        kept.append(stack[j])
        last = d_all[j]
    if len(kept) < cfg['min_slices'] <= len(stack):
        step = stack[0]['sbs'] or stack[0]['thick'] or 1.0
        kept = [stack[j] for j in np.lexsort((np.array([f['index'] for f in stack]), np.array([f['path'] for f in stack]),
                                              np.array([f['inst'] for f in stack])))]
        base = kept[0]['ipp'].copy()
        for rank, f in enumerate(kept):
            f['ipp'] = base + rank * step * n
        src['ipp'] = 'synthesized'
    if len(kept) < cfg['min_slices']:
        raise ValueError(f'{series_dir}: only {len(kept)} distinct slices in main stack')
    dk = np.array([np.dot(f['ipp'], n) for f in kept])
    gaps = np.diff(dk)
    n_gap_dropped = 0
    if gaps.size > 2:
        med = float(np.median(gaps))
        breaks = np.nonzero(gaps > max(cfg['gap_break'] * med, med + 1.0))[0]
        if breaks.size:
            edges = np.concatenate([[0], breaks + 1, [len(kept)]])
            runs = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
            a, b = max(runs, key=lambda r: r[1] - r[0])
            n_gap_dropped = len(kept) - (b - a)
            kept = kept[a:b]
            dk = dk[a:b]
            gaps = np.diff(dk)
    if len(kept) < cfg['min_slices']:
        raise ValueError(f'{series_dir}: only {len(kept)} contiguous slices in main stack')
    ipp = np.array([f['ipp'] for f in kept])
    ok = ipp - dk[:, None] * n
    ps = kept[0]['ps']
    modality = str(first.get('Modality', '')).upper()
    plan = dict(
        series_dir=str(series_dir),
        files=[(f['path'], f['index'], f['multi'], f['slope'], f['inter']) for f in kept],
        d=dk, o=ok, u=u, v=v, n=n, dr=float(ps[0]), dc=float(ps[1]),
        rows=kept[0]['rows'], cols=kept[0]['cols'],
        modality=modality,
        photometric=str(first.get('PhotometricInterpretation', 'MONOCHROME2')),
        samples=int(first.get('SamplesPerPixel', 1) or 1),
    )
    summary = dict(
        n_files=len(paths), n_frames=len(frames), n_stacks=len(stacks), stack_frames=len(stack),
        n_slices=len(kept), n_duplicates=len(stack) - len(kept) - n_gap_dropped, n_gap_dropped=n_gap_dropped,
        multiframe=bool(kept[0]['multi']), rows=plan['rows'], cols=plan['cols'],
        dr=plan['dr'], dc=plan['dc'], gap_median=float(np.median(gaps)), gap_min=float(gaps.min()),
        gap_max=float(gaps.max()), thickness=kept[0]['thick'], plane=dominant_plane(n),
        oblique_deg=float(np.degrees(np.arccos(min(1.0, np.abs(n).max())))),
        src_iop=src['iop'], src_ps=src['ps'], src_ipp=src['ipp'], modality_tag=modality,
        sop_class=str(first.get('SOPClassUID', '')),
        transfer_syntax=str(first.file_meta.get('TransferSyntaxUID', '')),
        photometric=plan['photometric'],
    )
    world = []
    if points:
        index = {}
        for f in frames:
            index[(f['sop'], f['index'] if f['multi'] else 0)] = f
        for pt in points:
            key = (pt['sop'], int(pt['f']) if pt.get('f') is not None else 0)
            f = index.get(key)
            if f is None:
                raise KeyError(f'{series_dir}: localizer SOP/frame {key} not found')
            if f['multi'] and pt.get('f') is None and frame_count_of(frames, pt['sop']) > 1:
                raise ValueError(f'{series_dir}: multi-frame localizer without frame index')
            world.append(f['ipp'] + pt['x'] * f['ps'][1] * f['u'] + pt['y'] * f['ps'][0] * f['v'])
    plan['points_world'] = np.array(world).reshape(-1, 3)
    return plan, summary


def implausible_fraction(h, keep):
    return float(np.mean(h[keep] < -1100.0)) if keep.any() else 0.0


def padding_free(bare, gap=50.0, max_values=3):
    vals, cnt = np.unique(bare, return_counts=True)
    keep = np.ones(bare.shape, bool)
    for i in np.argsort(-cnt)[:max_values]:
        if cnt[i] <= 0.05 * bare.size:
            break
        near = np.abs(vals - vals[i]) <= gap
        near[i] = False
        if cnt[near].sum() >= 0.01 * cnt[i]:
            break
        keep &= bare != vals[i]
    return keep


def stale_ct_intercept(bare, inters, margin=0.05):
    if not np.any(inters):
        return False
    keep = padding_free(bare)
    return implausible_fraction(bare + inters, keep) - implausible_fraction(bare, keep) > margin


def load_volume(plan, cfg, pixels=None):
    K = len(plan['files'])
    vol = np.empty((K, plan['rows'], plan['cols']), np.float32)
    groups = {}
    for slot, (path, idx, multi, slope, inter) in enumerate(plan['files']):
        groups.setdefault(path, []).append((slot, idx, slope, inter))
    for path, items in groups.items():
        px, nf = read_pixels(path, cfg) if pixels is None else pixels[path]
        if plan['samples'] > 1:
            px = px.astype(np.float32).mean(-1)
        if px.ndim == 2:
            px = px[None]
        if px.shape[0] != nf:
            raise ValueError(f'{path}: decoded {px.shape[0]} frames, expected {nf}')
        for slot, idx, slope, inter in items:
            vol[slot] = px[idx].astype(np.float32) * slope + inter
    if plan['modality'] == 'CT':
        inters = np.array([f[4] for f in plan['files']], np.float32)
        k = np.unique(np.linspace(0, K - 1, min(24, K)).round().astype(int))
        bare = vol[k][:, ::4, ::4] - inters[k][:, None, None]
        if stale_ct_intercept(bare, np.broadcast_to(inters[k][:, None, None], bare.shape)):
            vol -= inters[:, None, None]
            plan['ct_intercept'] = 'ignored'
        else:
            plan['ct_intercept'] = 'applied'
    if plan['photometric'] == 'MONOCHROME1':
        vol = vol.max() + vol.min() - vol
    return vol


def load_series(series_dir, cfg, pool, points=None):
    paths = list_series_files(series_dir)
    loaded = list(pool.map(read_full, paths, [cfg] * len(paths)))
    frames = [f for fr, _, _ in loaded for f in fr]
    first = read_dicom(paths[0], cfg['strict'], False)
    pixels = {p: (px, nf) for p, (_, px, nf) in zip(paths, loaded)}
    plan, summary = build_plan(series_dir, cfg, points=points, headers=(frames, first))
    return plan, summary, load_volume(plan, cfg, pixels=pixels)


def multiframe_spacing_table(series_dirs):
    table = {}
    for sd in series_dirs:
        files = list_series_files(sd)
        if len(files) != 1:
            continue
        ds = pydicom.dcmread(files[0], stop_before_pixels=True)
        if frame_count(ds) < 2:
            continue
        frames = frames_of(files[0], ds)
        step = position_step(frames)
        ps = frames[0]['ps']
        if step is None or ps is None:
            continue
        key = table_key(frames[0]['rows'], frames[0]['cols'], step)
        table.setdefault(key, []).append([float(ps[0]), float(ps[1])])
    return {k: dict(ps=np.median(np.array(v), axis=0).tolist(), count=len(v)) for k, v in table.items()}


def save_table(table, path):
    with open(path, 'w') as f:
        json.dump(table, f, indent=1, sort_keys=True)


def load_table(path):
    with open(path) as f:
        return json.load(f)
