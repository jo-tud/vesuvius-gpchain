#!/usr/bin/env python3
"""Organiser segment mesh (tifxyz on the 2.4 um scan) -> the same mesh in level-2 voxels (9.6 um), the frame of the
fit and of the CT that gpchain renders. Used as the layer-order reference (chain.py --segments); no label is read.

Rule (unchanged from the stored runs): keep every 4th node in rows and columns, so a node stays 20 canvas px at
level 2, and map xyz with the pyramid voxel-centre rule p2 = (p0 - 1.5) / 4 (level-2 voxel k is centred on level-0
coordinate 4k + 1.5). Holes (-1) stay holes. Writes x.tif, y.tif, z.tif (float32), no meta.json.

  python segments_to_l2.py SRC.tifxyz DST_DIR        (run in the w2401 venv: numpy, tifffile)
"""
import sys
from pathlib import Path

import numpy as np
import tifffile

F = 4


def to_level2(g0):
    """(3, R, C) level-0 xyz grid -> every 4th node, level-2 voxel coords; holes stay -1."""
    g = g0[:, ::F, ::F].astype(np.float64)
    hole = np.all(g == -1, 0) | ~np.all(np.isfinite(g), 0)
    out = (g - (F - 1) / 2) / F
    out[:, hole] = -1
    return out.astype(np.float32)


def main(src, dst):
    src, dst = Path(src), Path(dst)
    g2 = to_level2(np.stack([tifffile.imread(src / f'{c}.tif') for c in 'xyz']))
    dst.mkdir(parents=True, exist_ok=True)
    for k, c in enumerate('xyz'):
        tifffile.imwrite(dst / f'{c}.tif', g2[k])
    print(dst, list(g2.shape[1:]))


if __name__ == '__main__':
    main(*sys.argv[1:3])
