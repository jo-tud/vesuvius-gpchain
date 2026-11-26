# Provenance

Source: the authors' working repository. "Run commit" = `7746852a`, the commit of the fresh-clone repeat whose
numbers are in `results/f637_fresh_clone_repeat_6_windings.json`; `tools/gpchain` there equals `tools/gpchain` at
the stored F-626 run for every file the chain and the ink step use.

| file | source | change |
|---|---|---|
| `tools/gpchain/*.py`, `*.json`, `templates/`, `tests/` (except the two below) | `tools/gpchain/`, run commit | identical |
| `tools/gpchain/package.py` | same path, run commit | 1 line: default of `--store` `/home/…/kept` → `kept` (a host path) |
| `tools/gpchain/records.py` | same path, run commit | 1 line: the same default |
| `tools/gpchain/orient_measure.py`, `tools/gpchain/CHECKLIST.md` | same path | left out: the first imports experiment code outside this tree; the second is an internal rule checklist |
| `scripts/docker-shim.py` | `docker/gpchain/docker-shim.py`, run commit | the bundle folder comes from `GPCHAIN_VCB` (default `/opt/vcb` as before); shebang and docstring |
| `scripts/bundle.sh` | `docker/gpchain/bundle.sh`, run commit | identical |
| `build/vc3d-47111d1a/Dockerfile` | the `vc3d` stage of the repository's root `Dockerfile` (unchanged since commit `b54f82e`, which built the render binary of the stored runs) | the stage only, as its own file; header comment; `COPY` path of `apt-install.sh` |
| `build/vc3d-47111d1a/apt-install.sh` | `docker/apt-install.sh` | identical except comment line 7 (our host name replaced by "our build host") |
| `build/villa-6bbe6e21/Dockerfile` | `tools/docker/villa-ink-6bbe6e2/Dockerfile` at commit `92fba3b9` (built the flattening binaries of the stored runs) | the `base` and `vcbuild` stages unchanged; its Python, lasagna and model stages left out; a `tools` stage copies the built programs and villa's `LICENSE` |
| `env/requirements-cpu.txt`, `env/requirements-gpu.txt` | `experiments/2026-10-17-9um-power-map/runtime/` | identical |
| `env/requirements-w2401.txt` | `experiments/2026-11-08-ves3560-venv-rebuild/` | identical |
| `scripts/make_venvs.sh` | `docker/gpchain/venvs.sh` | same uv commands and pins; host folders instead of `/opt`; `cpu` optional |
| `scripts/fetch_ct.py` | the fetch step of the stored runs (`touched()` with a 24-voxel pad, chunk download) | rewritten as one script; the hard-link store left out |
| `scripts/segments_to_l2.py` | the `to_level2` step that made the stored level-2 segment meshes | the same function; output byte-equal (checked on all 4 segments) |
| `scripts/coverage.py` | the coverage rule of the stored band run (F-626) | rewritten as one script; on the stored outputs it gives 120 of 120, as stored |
| `scripts/make_folder.py` | the `images` step of the stored band runs | rewritten as one script |
| `scripts/build_villa.sh`, `scripts/get_inputs.py`, `scripts/run_winding.sh` | new | new |
| `inputs/manifest.json` | sha256 of every input file (the fit's 120 windings, 4 segments, umbilicus, models, CT source) | new |
| `results/f610_dev_band_z11000-12000.json` | `experiments/2026-11-09-ves3574-gpchain-fullband/attempt4/results.json` | identical |
| `results/f623_fail_first_heldout_read_z12000-13000.json` | `experiments/2026-11-10-ves3833-gpchain-heldout/results.json` | identical |
| `results/f626_heldout_band_z13000-14000.json` | `experiments/2026-11-11-ves3868-rim-band/results.json` | identical |
| `results/f637_fresh_clone_repeat_6_windings.json` | `experiments/2026-11-13-ves3938-gpchain-repeat/results.json` | identical |
| `results/f622_fail_column_finder.json` | `experiments/2026-11-10-ves3775-gen19-heldout/attempt4/results.json` | one host path prefix → `<data>/` |
| `results/f644_outside_path_check_3_windings.json` | `experiments/2026-11-13-ves3952-gpchain-release/results.json` (comparator v4, after the R9 fixes of VES-3999, VES-4001 and VES-4003) | the run folder path → `NEW` |
| `figure.png`, `results/figure_w037_column_01.json` | made by `experiments/2026-11-13-ves4053-gpchain-fixes/make_figure.py` (the VES-3952 script plus a held-back mask near internal test pages: 5 vx from a page vertex or a 16-layer render slab within 3 vx (half a sheet thickness) of a page surface, dilated 3 px; 2.0 % of the piece; VES-4094, VES-4106) from the stored F-626 outputs (it reads our internal store, so it is not in this tree) | new |

Comments and result files keep internal task and finding ids (`VES-…`, `F-…`); they name our records and hold no
paths. The F-ids in the result file names: F-610 dev band, F-623 first held-out read (fail), F-626 held-out band,
F-637 fresh-clone repeat, F-622 column finder (fail).
