import pickle

import imagecodecs
import numpy as np
import zstandard


def encode_volume(img, dist, seg=None, level=10):
    z = zstandard.ZstdCompressor(level=level)
    rec = dict(
        version=1, shape=tuple(img.shape),
        img=[imagecodecs.jpegls_encode(np.ascontiguousarray(img[i])) for i in range(img.shape[0])],
        dist=z.compress(np.ascontiguousarray(dist).tobytes()),
        seg=None if seg is None else z.compress(np.ascontiguousarray(seg).tobytes()),
    )
    return pickle.dumps(rec, protocol=pickle.HIGHEST_PROTOCOL)


def read_volume(path, parts=('img', 'dist', 'seg')):
    with open(path, 'rb') as f:
        rec = pickle.load(f)
    if rec.get('version') != 1:
        raise ValueError(f'{path}: unsupported volume record version {rec.get("version")}')
    shape = tuple(rec['shape'])
    out = {}
    if 'img' in parts:
        out['img'] = np.stack([imagecodecs.jpegls_decode(b) for b in rec['img']])
        if out['img'].shape != shape:
            raise ValueError(f'{path}: decoded image {out["img"].shape} != stored shape {shape}')
    d = zstandard.ZstdDecompressor()
    if 'dist' in parts:
        out['dist'] = np.frombuffer(d.decompress(rec['dist']), np.uint8).reshape(shape)
    if 'seg' in parts:
        out['seg'] = None if rec['seg'] is None else np.frombuffer(d.decompress(rec['seg']), np.uint8).reshape(shape)
    return out
