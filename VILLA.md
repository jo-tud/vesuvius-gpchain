# villa in this chain: pins, licences, patches

gpchain runs villa ([github.com/ScrollPrize/villa](https://github.com/ScrollPrize/villa)) code at three pinned
commits. **No villa source file is patched.** Nothing in this tree is copied from villa; the build recipes fetch villa
from GitHub at the commit and check the hash.

| villa commit | what gpchain uses | where it runs | villa licence of that part |
|---|---|---|---|
| `6bbe6e211c56079c763440105b299463a0c8e6d6` | VC3D tools `vc_tifxyz2obj`, `flatboi`, `vc_obj2tifxyz` (isometric stage, band cuts) | built by `build/villa-6bbe6e21/Dockerfile` | `volume-cartographer/`: **GPL-3.0 or later** (Copyright 2023 EduceLab; `volume-cartographer/LICENSE`, `NOTICE`) |
| `6bbe6e211c56079c763440105b299463a0c8e6d6` | `spiral-fitting/` (`fit_spiral.py --dataset`): made the fit that is the chain's input | not part of this tree (the fit is an input) | villa root `LICENSE`: MIT (Copyright 2024 Vesuvius Challenge) |
| `47111d1a31381a5ff2c05046ed3e311b0dbeec64` | VC3D `vc_render_tifxyz` (16-layer render of each piece) | built by `build/vc3d-47111d1a/Dockerfile` | `volume-cartographer/`: GPL-3.0 or later, as above |
| `47111d1a31381a5ff2c05046ed3e311b0dbeec64` | Python package `vesuvius` (in the `gpu` / `cpu` environments) | `scripts/make_venvs.sh` | MIT (villa root `LICENSE`; `vesuvius/pyproject.toml` classifier MIT) |
| `ea48ce6ba32cbb854c292f42bc2ad3286fafc1b4` | Python package `vesuvius` with `vesuvius.gp_submission` (the Grand Prize format check of pull request #2401, closed, not merged) | `scripts/make_venvs.sh` (`w2401` environment) | MIT, as above |

## Upstream status (checked 2026-11-26)

- `ea48ce6b` is the head of **villa pull request #2401** ("vesuvius: add gp_submission, cut Grand Prize column meshes
  and check the folder"), a draft pull request by phippsrtyler. It was **closed by its author on 2026-11-24, not
  merged**: `vesuvius.gp_submission` is not in villa and is not an organiser tool. The check we pass is that pull
  request's check at this commit. `scripts/make_venvs.sh` fetches the commit by its hash; GitHub still served it on
  2026-11-26, but it may stop serving the commits of a closed pull request. If a Grand Prize check appears in villa
  later, our result has to be checked against it again.
- `6bbe6e21` and `47111d1a` are commits on villa's main history. Newer villa versions are **not tested** with gpchain.
  Two things in this chain depend on villa behaviour: flatboi's flattening (the isometry gate of the check) and the
  no-data rim of `vc_render_tifxyz` (villa leaves the outer half node cell of a piece unrendered; the edge-band frame
  check is sized to it). A villa change in either can change our results.

## Our changes

| what | change | upstream status |
|---|---|---|
| villa source | none | – |
| villa build scripts | `volume-cartographer/scripts/install_build_deps.sh` (6bbe6e21): its AWS CLI block is cut by `sed` inside the build (unpinned download, not needed to build) | not proposed upstream (build-environment choice, not a fix) |
| VC3D CMake deps (47111d1a build) | libbacktrace and vc-delta3d are fetched at fixed commits instead of a moving branch / tag, so a rebuild cannot pick up new code | not proposed upstream |

## Why two VC3D commits

The stored runs used two local builds made for earlier work: the flattening tools at `6bbe6e21` (F-429, the isometric
stage) and the render at `47111d1a` (F-576, the render contract). All stored numbers come from that pair, so the
release keeps it. A single newer VC3D build is the obvious simplification; it is not tested.

## Licence consequence for this tree

The code in this tree is MIT (`LICENSE`). It calls the VC3D tools as separate programs; it does not copy, link or ship
them. Whoever builds them with `scripts/build_villa.sh` gets GPL-3.0 programs built from villa's source. A published
binary bundle or Docker image with these programs would have to meet GPL-3.0 (corresponding source offered); this
release publishes none.
