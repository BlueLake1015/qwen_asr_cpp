# offline/ — air-gapped deployment

Build and run `qwen_asr_cpp` on a machine with **no internet**. The flow mirrors a
standard apt offline bundle: gather `.deb`s on an online host, install on the
target.

```
ONLINE host (matching Ubuntu + arch)            OFFLINE target
─────────────────────────────────────           ──────────────────────────────
offline/download_debs.sh  ──▶ offline/debs/  ─(copy whole repo)─▶  offline/install.sh ──▶ make
                                                                    + copy models/  ──▶  build/qwen-asr
```

## 1. On an ONLINE machine (same Ubuntu release + CPU arch as the target)

```bash
offline/download_debs.sh        # fills offline/debs/ with the full dependency closure
```

Bundles build tools (`g++`, `g++-12`, `gcc-12`, `make`, `cmake`), `ffmpeg`,
`git`/`curl`, and the **CUDA toolkit** (`nvidia-cuda-toolkit` → `nvcc` + cuBLAS/
cuDART), plus every recursive dependency. Writes `manifest.txt` and `target.txt`
(the release/arch the bundle targets).

> The NVIDIA **driver** is *not* bundled — the GPU target is assumed to already
> have a working driver (`nvidia-smi` runs). Driver installs need a matching
> kernel module + reboot; handle that out of band, or build CPU-only (`make GPU=off`).

## 2. Transfer to the target

Copy the **entire repo** — crucially with the submodule populated
(`third_party/CrispASR/` and its in-tree `ggml/`, which ship checked out and
already carry the `--aligned` patch) — **and** `offline/debs/`. e.g.:

```bash
tar czf qwen_asr_cpp_offline.tgz qwen_asr_cpp/    # includes offline/debs/ + third_party/
# move the tarball to the offline machine, extract
```

Models are large and separate: run `scripts/get_model.sh` on an online machine and
copy the resulting `models/` directory over too (the app loads only from `models/`).

## 3. On the OFFLINE target

```bash
offline/install.sh    # dpkg-installs the bundled .debs, verifies the toolchain
make                  # builds the CUDA engine + app (or: make GPU=off)
build/qwen-asr tests/fixtures/test_en_30s.mp4 -o out.srt -l en
```

`install.sh` warns if the host's Ubuntu/arch doesn't match the bundle, runs two
`dpkg -i` passes (self-contained closure, no downloads), and checks for
`g++/g++-12/make/cmake/ffmpeg/nvcc` + the GPU driver.

## Notes

- `offline/debs/` is git-ignored (large). Re-create it with `download_debs.sh`.
- Release/arch must match: `.deb`s built for 24.04/amd64 won't install on 22.04 or arm64.
- No Python is involved anywhere in build, install, or runtime.
