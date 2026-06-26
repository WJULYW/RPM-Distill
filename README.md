# RPM-Distill: Physiology-guided Adaptive Cross-modal Distillation for Robust Remote Physiological Measurement

This is the official implementation of our paper **"RPM-Distill: Physiology-guided Adaptive Cross-modal Distillation for Robust Remote Physiological Measurement"**.

> Video-based remote physiological measurement (RPM) is highly accessible but remains fragile under varying illumination, skin tones, and motion. Radio-frequency (RF) radar is largely invariant to illumination and appearance, providing complementary cardio-respiratory micro-motion cues; however, requiring radar at inference is often impractical due to its limited ubiquity and deployment overhead. We propose **RPM-Distill**, a physiology-guided cross-modal distillation framework that leverages synchronized radar **only during training** while retaining **video-only inference**. Although RGB and RF waveforms differ in sensing physics and time-domain morphology, they share a similar latent periodic rhythm in the frequency domain. We thus distill physiology-structured spectral evidence through three complementary losses that (i) anchor the fundamental peak, (ii) match the off-peak background distribution, and (iii) preserve spectral morphology and sharpness. To avoid negative transfer under sample-level teacher quality and alignment uncertainty, a **spectral policy network** predicts per-sample distillation gates and component weights from the student–teacher spectral relation map, learned with a **bilevel meta objective** on a small labeled validation split.

If you use this code, please kindly cite our work (see [Citation](#citation)).

---

## Overview

RPM-Distill follows a teacher–student paradigm:

- **Student** — a video-based rPPG network (default: `FactorizePhys`) that predicts the BVP waveform from RGB clips. **This is the only model needed at inference.**
- **Teacher** — a frozen, pre-trained RF radar network (default: `RF_conv_decoder`) that provides privileged spectral guidance **only during training**.
- **Spectral Policy Network** — a 1D-conv encoder with a matrix-decomposition decoder that consumes the three-channel spectral relation map `[ℓ_v, ℓ_r, |ℓ_v − ℓ_r|]` and outputs a per-sample gate `g ∈ (0,1)` and component weights `α ∈ Δ²` over `{peak, off, shape}`.

Training interleaves three steps per meta-interval (bilevel meta-optimization):

1. **Virtual student update** on `D_tr` under the current policy.
2. **Policy update** on a held-out labeled validation split `D_val`.
3. **Real student update** on `D_tr` with the refined policy.

The three physiology-structured distillation components correspond in the code to `L_band_peak` (fundamental peak alignment), `L_band_other` (background noise suppression), and `L_struct` (spectral morphology / sharpness consistency).

---

## Repository Structure

```
Code/
├── main.py                       # Entry point: training + evaluation pipeline
├── requirements.txt
├── data/
│   ├── dataset_factory.py        # Builds paired RGB/RF/PPG datasets
│   ├── equipleth_dataset.py      # Paired video–RF window dataset with strict alignment
│   ├── organizer.py              # Raw radar frame organizer
│   └── rf_processing.py          # Range–time / IQ radar pre-processing
├── models/
│   ├── model_selector.py         # get_model(modality, name)
│   ├── rgb/FactorizePhys.py      # Video student backbone
│   └── rf/RF_conv_decoder.py     # RF teacher backbone
├── losses/
│   ├── NegPearsonLoss.py         # Negative Pearson correlation loss
│   └── SNRLoss.py                # SNR loss in the physiological band
└── utils/
    ├── checkpoint_manager.py     # Hierarchical checkpoint save/load
    ├── gpu_utils.py              # Automatic GPU selection (CPU fallback)
    └── utils.py                  # HR-from-PSD, detrending, helpers
```

The spectral distillation losses, the policy network (`PolicyConvNet` / `MatrixDecompositionDecoder`), and the bilevel meta-optimization loop are all implemented in [`main.py`](main.py).

---

## Requirements

```bash
conda create -n rpm-distill python=3.9 -y
conda activate rpm-distill
pip install -r requirements.txt
```

The code is implemented in PyTorch and was tested on a single NVIDIA GPU (the paper uses an A800). `pynvml` is optional and only used for automatic GPU selection; if it is absent the code falls back to `cuda:0` or CPU.

---

## Data Preparation

Following the paper, the framework is trained on synchronized **RGB video + RF radar** datasets (e.g., **EquiPleth**, **PhysDrive**) and can be cross-evaluated on RGB-only datasets (**PURE**, **MMPD**). Please obtain each dataset from its official source and accept the corresponding agreements.

This release ships a paired-window data interface (`PairedVideoRFWindowDataset`) that expects each recording to be organized as:

```
<data_dir>/
├── rgb_files/<video_name>/
│   ├── rgbd_rgb_0.png, rgbd_rgb_1.png, ...   # extracted, face-cropped frames
│   └── rgbd_ppg.npy                          # ground-truth BVP/PPG (optional per sample)
└── rf_files/<rf_folder>/
    ├── rf.pkl                                # raw radar cube
    └── vital_dict.npy                        # vital signals (fallback PPG source)
```

A `*.pkl` **fold file** maps fold index → `{"train": [...], "val": [...], "test": [...]}` lists of `video_name`s. Point the pipeline at your data with:

- `--data-dir` — root that contains `rgb_files/` and `rf_files/`
- `--folds-path` — path to the `*.pkl` fold split

> Default sampling rate is **30 Hz**, the physiological band is **[45, 180] bpm**, each clip is **T = 256** frames with a stride of **30**, and frames are face-cropped and resized. These can be changed via the corresponding CLI flags.

---

## Pre-trained Models (RF Teacher)

The RF teacher is **pre-trained and frozen** during distillation. Train (or download) an `RF_conv_decoder` teacher and place its checkpoint so that the `CheckpointManager` can find it under:

```
<checkpoints_path>/<dataset>/rf/RF_conv_decoder/frame<frame_length>_fold<fold>_step<step>/best_model.pth
```

You may also pass the teacher checkpoint explicitly with `--rf-checkpoint /path/to/best_model.pth`. The student is trained from scratch by default (the student-checkpoint warm-start is commented out in `main.py`).

> `--checkpoints_path` is **required** — it is the root used both for locating the teacher and for saving student/policy checkpoints during training.

---

## Training and Testing

Training and evaluation are run together by `main.py`: the script first reports a pre-training baseline, trains the student with the meta-distillation loop, and finally evaluates the student and writes a CSV to `eval_results/`.

Basic command:

```bash
python main.py \
    -s equipleth -t equipleth \
    --data-dir   /path/to/data \
    --folds-path /path/to/folds.pkl \
    --checkpoints_path /path/to/ckpt \
    --rf-checkpoint    /path/to/rf/best_model.pth \
    --student-model FactorizePhys \
    --teacher-model RF_conv_decoder \
    --train-mode sup_plus_distill \
    --epochs 5 --frame-length 256 --step 30 --fs 30 \
    --learning-rate 5e-5 --weight-decay 1e-2 \
    --lambda-distill 1.0 --meta-lr 1e-4 --meta-interval 10 \
    --device 0
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `-s`, `--source-domain` / `-t`, `--target-domain` | `equipleth` | Source (train) and target (test) domains. Equal → intra-dataset split; different → cross-dataset. |
| `--checkpoints_path` | — | **(Required)** Root for loading the teacher and saving checkpoints. |
| `--rf-checkpoint` | auto | Path to the frozen RF teacher checkpoint. |
| `--student-model` / `--teacher-model` | `FactorizePhys` / `RF_conv_decoder` | Student and teacher backbones. |
| `--train-mode` | `sup_plus_distill` | `sup_only`, `distill_only`, or `sup_plus_distill`. |
| `--lambda-distill` | `1.0` | Weight of the distillation term `λ_distill`. |
| `--meta-lr` | `1e-4` | Policy (meta) learning rate `β`. |
| `--meta-interval` | `10` | Run a bilevel meta-update every N student steps. |
| `--virtual-lr` | = `learning-rate` | Virtual student step size `η` for the meta update. |
| `--gate-reg` | `0.0` | Optional regularizer encouraging the distillation gate to stay open. |
| `--policy-k` | `3` | Number of distillation components — **must be 3** (peak / off / shape). |
| `--policy-md-rank` / `--policy-md-hidden` | `8` / `32` | Matrix-decomposition decoder rank / hidden width. |
| `--frame-length` / `--step` / `--fs` | `256` / `30` / `30` | Clip length, stride, sampling rate. |
| `--l-freq-bpm` / `--u-freq-bpm` | `45` / `180` | Physiological band for all spectral operations. |
| `--device` | `cuda` | GPU index (e.g., `0`) or `cpu`. |

### Ablations

The three-component objective and the policy can be toggled directly:

- **Distillation only / supervised only:** `--train-mode distill_only` or `--train-mode sup_only`.
- **Disable the gate regularizer / tune the gate:** `--gate-reg 0.0`.
- **Disable the policy input gradient:** the policy input is detached by default; use `--no-policy-detach-input` to keep gradients.

### Output

Evaluation metrics (HR **MAE / RMSE / r / STD** in bpm, plus SNR and spectral peak/band-ratio errors) are printed to stdout and appended to:

```
eval_results/{intra|cross}/<target>/RPM-Distill/<student-model>/frame<...>_fold<...>_step<...>/result.csv
```

---

## Citation

If our work is helpful to your research, please consider citing:

```bibtex
@inproceedings{wang2026rpmdistill,
  title     = {RPM-Distill: Physiology-guided Adaptive Cross-modal Distillation for Robust Remote Physiological Measurement},
  author    = {Wang, Jiyao and Hu, Qingyong and Tang, Duoxun and Yang, Xiao and Wu, Kaishun and Yu, Jiangbo},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## Acknowledgements

The video student backbone and the radar pre-processing pipeline build upon prior open-source rPPG/RF works (e.g., FactorizePhys and EquiPleth). The Negative Pearson and SNR losses follow the implementations of Yu et al. (BMVC 2019) and related rPPG literature. We thank the authors of these works for releasing their code.
