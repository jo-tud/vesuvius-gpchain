# Third-party material and the licences of the inputs

Checked 2026-11-26. Nothing below forbids this release. The code in this tree is MIT (`LICENSE`); none of the
material below is copied into the tree (the only exception is the hash-locked package lists in `env/`, which name
packages and do not contain them).

## Programs and code the chain calls

| what | source | licence | how it is used here |
|---|---|---|---|
| VC3D tools `vc_tifxyz2obj`, `flatboi`, `vc_obj2tifxyz` | villa `volume-cartographer/` at `6bbe6e21` | **GPL-3.0 or later** (Copyright 2023 EduceLab; `volume-cartographer/LICENSE`, `NOTICE`) | built from source by `scripts/build_villa.sh`, called as separate programs; not shipped |
| VC3D `vc_render_tifxyz` | villa `volume-cartographer/` at `47111d1a` | GPL-3.0 or later, as above | as above |
| `vesuvius` Python package incl. `vesuvius.gp_submission` (the Grand Prize folder check) | villa `vesuvius/` at `47111d1a` and at the head `ea48ce6b` of pull request #2401 (a draft by phippsrtyler, closed by its author 2026-11-24, not merged) | MIT (villa root `LICENSE`, Copyright 2024 Vesuvius Challenge) | installed by `scripts/make_venvs.sh` from GitHub; not shipped |
| hecate model code `hecate.py` | Hugging Face `scrollprize/hecate`, revision `9cb86e50` | MIT (Copyright 2026 Vesuvius Challenge) | downloaded by `scripts/get_inputs.py`, sha256 checked; the ink model architecture |
| Python packages (numpy, scipy, torch, tifffile, zarr, ...) | PyPI, download.pytorch.org; exact versions and hashes in `env/requirements-*.txt` | each its own (BSD, MIT, Apache-2.0, ...) | installed by `scripts/make_venvs.sh`; not shipped |
| base images in `build/` | `ubuntu:26.04`, `debian:trixie-20261005-slim` (by digest), Ubuntu / Debian snapshot archives | their distributions' licences | only used to build the VC3D tools |

villa pins, upstream status and the list of our changes (none to villa source): `VILLA.md`.

## Models

| what | source | licence | how it is used here |
|---|---|---|---|
| frozen hecate weights `hecate_9.6um.pth` | Hugging Face `scrollprize/hecate`, revision `9cb86e50`, sha256 `809f4f10…fcfe5d` | MIT | the base of S; downloaded only with `get_inputs.py --frozen` |
| **S** checkpoint (`hecate_ft.pth`, sha256 `4b84fdf2…05b4f2`, 523 MB) | our fine-tune of `hecate_9.6um.pth`, released separately: https://huggingface.co/jo-tud/hecate-onsheet (file `hecate_onsheet_S.pth`) | CC BY-NC 4.0, hecate's MIT notice kept (it was trained on CC BY-NC 4.0 labels) | the ink model of every map here; not in this tree |

## Data

| what | source | terms | how it is used here |
|---|---|---|---|
| PHerc. Paris 4 CT, scan `20260411134726` (2.4 µm, 78 keV), pyramid level 2 (9.6 µm) | `s3://vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr` | CC BY-NC 4.0 (`LICENSE.txt` at the bucket root) | `scripts/fetch_ct.py` reads the chunks a mesh touches over https; not in this tree |
| PHerc. Paris 4 segment meshes `20230929220926`, `20231012184424`, `20231106155351`, `20231210121321` | `.../PHercParis4/segments/<id>/mesh/<id>-on-20260411134726-2.4um.tifxyz/` | CC BY-NC 4.0 | geometry only (layer-order reference); converted to level 2 by `scripts/segments_to_l2.py`; no ink label is read |
| Paris 4 umbilicus (146 points) | `https://dl.ash2txt.org/datasets/spiral_datasets/PHercParis4/umbilicus.json` (level-2 coordinates, the file the chain reads); the same points × 4 are `s3://vesuvius-challenge-open-data/PHercParis4/representations/umbilicus/20260411134726-umbilicus-20260524235033.json` | CC BY-NC 4.0 (bucket copy); the data server's terms apply to the server copy (below) | layer-order reference; downloaded by `get_inputs.py`, not in this tree |
| **our fit** `T-B-s2` (120 per-winding meshes, z 13000-14000) | made by us with villa `spiral-fitting` at `6bbe6e21` from the organisers' Paris 4 spiral dataset (tracks, normals, gradient magnitude, outer shell; data server `spiral_datasets/PHercParis4`, also on the bucket under `representations/spiral/`) and the CT; no ink label, no human winding input | our result; CC BY-NC 4.0 like its inputs | the chain's input; a release asset of this repository (`inputs/manifest.json` lists every file and its sha256) |

**Data server terms** (`https://dl.ash2txt.org/LICENSE.txt`, sha256 `9dc08a92…8d154`): no redistribution of the
data without written approval, and no public revelation of hidden text outside the official Discord without
consent. This tree redistributes no organiser data: every organiser file is downloaded by the user from the
organisers. Our fit is a result computed from their data, not a copy of it.

**PHerc. Paris 4:** the organisers said on 2026-11-09 that Paris 4 results and images may appear in progress-prize
entries and public repositories; academic publication of Paris 4 results in journals or conferences is not allowed
before 2027-06-25. PHerc. Paris 4 is **not** one of the volumes eligible for the 2027 Grand Prize; it is the scroll the
chain was developed and tested on.

## Credits

- **villa / Volume Cartographer (VC3D)**: Vesuvius Challenge and EduceLab, for the tools above, the tifxyz format,
  and spiral fitting.
- **Grand Prize format check** (`vesuvius.gp_submission`): phippsrtyler, villa pull request #2401 (a draft, closed by
  its author on 2026-11-24, not merged).
- **hecate**: the ink model of the Vesuvius Challenge (`scrollprize/hecate`), the base of our S checkpoint.
- **CT scans, segment meshes, spiral dataset and umbilicus**: Vesuvius Challenge open data.

## hecate licence text

```
MIT License

Copyright (c) 2026 Vesuvius Challenge

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
