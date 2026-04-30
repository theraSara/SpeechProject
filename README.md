# Few-Shot Open-Set Learning for On-Device Customization of KWS

This repository contains two complementary approaches for few-shot open-set keyword spotting on the Google Speech Commands dataset:

| Folder | Approach |
|--------|----------|
| [`dl_branch/`](dl_branch/) | Deep learning: metric learning, knowledge distillation, SSL fine-tuning |
| [`traditional_branch/`](traditional_branch/) | Classical baselines: MFCC + DTW, MFCC + HMM-GMM |

---

## DL Branch

### Installation

```bash
pip install -r requirements.txt
```

If that fails, install the tricky dependencies first then retry:

```bash
conda install visdom
pip install torchnet

python -m pip install numpy cython
python -m pip install --no-build-isolation libmr

pip install -r requirements.txt
```

### Repository Overview
All the used scripts are included in the `KWSFSL/` folder:
* `metric_learning.py`: main training script for the student encoder
* `train_teacher.py`: fine-tuning script for SSL teacher models (Wav2Vec 2.0, HuBERT, WavLM) with LoRA
* `train_distill.py`: knowledge distillation training script (single or multi-teacher)
* `test_fewshots_classifiers_openset.py`: main test script, including the metric calculation
* `classifiers/`: classifiers used for the few-shot initialization (`ncm`, `openncm`, `ncm_openmax`)
* `models/`: collection of loss functions and backbones
* `scripts/`: data preparation utilities


### Data Preparation

Datasets are stored under a `data/` directory at the repo root. Also note that additive noise from the DEMAND dataset is used at training time.

#### Google Speech Commands (GSC)

```bash
mkdir -p data/GSC
cd data/GSC

wget -c http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
tar -xzf speech_commands_v0.02.tar.gz

cd ../..
```

#### Multilingual Spoken Words Corpus (MSWC)

```bash
mkdir -p data/MSWC
cd data/MSWC

wget -c -O en_audio.tar.gz  https://mswc.mlcommons-storage.org/audio/en.tar.gz
wget -c -O en_splits.tar.gz https://mswc.mlcommons-storage.org/splits/en.tar.gz


mkdir -p audio splits
tar -xzf en_audio.tar.gz -C audio
tar -xzf en_splits.tar.gz -C splits

cd ../..
```

After extraction, audio files will be in `data/MSWC/audio/en/clips/` and split CSVs (`en_train.csv`, `en_dev.csv`, `en_test.csv`, `en_splits.csv`) will be in `data/MSWC/splits/`.

**Convert audio to WAV (Optional).** MSWC distributes audio as `.opus` files. Convert them to `.wav` (PCM 16-bit) for faster loading at training time:

```bash
# Extract file paths from the full splits file
scripts/get-links data/MSWC/ en

# Convert to wav (64 parallel workers)
scripts/wav-convert data/MSWC/ en
```

**Compute word frequencies** (required for partition generation):

```bash
scripts/mswc-freq data/MSWC/
```

**Generate partition CSVs.** Use `mswc-sample` to select the training vocabulary. Key flags: `-n` selects the top-n most frequent words, `-b` sets a minimum file count per word (lower bound), `-t` sets a maximum file count per word (upper bound).

```bash
mkdir -p partitions
python scripts/mswc-sample data/MSWC/ train -l en -n <n> -b <b> -t <t> -go > partitions/train.csv
python scripts/mswc-sample data/MSWC/ dev  -l en -go > partitions/dev.csv
python scripts/mswc-sample data/MSWC/ test -l en -go > partitions/test.csv
```

**Noise data.** Add noise recordings to `data/MSWC/noise/`. We used samples from the [DEMAND](https://zenodo.org/record/1227121) dataset, copying the wav file with ID=01 of every noise type (the filename in the destination folder can be anything).


### Open-Set Test Framework with Few-Shot Example Enrollments

After [training](#feature-extractor-training), the feature extractor is evaluated on a few-shot open-set problem. _E.g._:
```bash
python KWSFSL/test_fewshots_classifiers_openset.py \
  --data.cuda \
  --speech.dataset googlespeechcommand \
  --speech.task GSC12,GSC22 \
  --speech.include_unknown \
  --fsl.test.n_way 11 \
  --fsl.test.n_episodes 10 \
  --speech.default_datadir data/GSC/ \
  --fsl.test.batch_size 264 \
  --fsl.classifier ncm \
  --fsl.test.n_support 5 \
  --model.model_path results/<MODEL_PATH>
```

#### Main Test Options
- `fsl.classifier`. Type of the classifier. Options: `ncm`, `openncm`, `ncm_openmax`, `dproto`.
- `fsl.test.n_support`. Number of support samples used to initialize the classifier (1, 5, or 10).
- `fsl.test.n_way`. Number of classes: N known + 1 `_unknown_`.
- `fsl.test.n_episodes`. Number of test episodes. At every episode, a different set of support samples is loaded.
- `model.model_path`. Path to the trained model (`.pt` file).


### Feature Extractor Training

To train the feature encoder on the MSWC dataset:
```bash
python KWSFSL/metric_learning.py \
  --data.cuda \
  --speech.dataset MSWC \
  --speech.task MSWC500U \
  --speech.default_datadir data/MSWC/audio/ \
  --speech.default_csvdir partitions/ \
  --speech.include_noise \
  --speech.noise_dir data/MSWC/noise/ \
  --model.model_name repr_conv \
  --model.encoding DSCNNL_LAYERNORM \
  --model.z_norm \
  --train.epochs 40 \
  --train.n_way 40 \
  --train.n_support 0 \
  --train.n_query 10 \
  --train.n_episodes 100 \
  --train.loss triplet \
  --train.margin 0.5 \
  --log.exp_dir results/<TEST_NAME>/
```

#### Main Training Options
- `speech.dataset`. Name of the dataset: `MSWC`.
- `speech.task`. `MSWC500U` means 500 classes with an unbalanced number of samples.
- `speech.include_noise`. If enabled, additive noise is added to the utterance samples.
- `speech.noise_dir`. Path to the directory containing noise `.wav` files (required when `--speech.include_noise` is set).
- `speech.use_wav`. If enabled, loads pre-converted `.wav` files from `clips_wav/` instead of `.opus` files from `clips/`. ONLY add this if you ran the WAV conversion step.
- `model.model_name`. `repr_conv` indicates the class of the model.
- `model.encoding`. Used encoder. `DSCNNL_LAYERNORM` is a DSCNN large model with a final layer norm layer.
- `model.z_norm`. If enabled, L2 normalization is applied on the embeddings.
- `train.n_way`. Number of classes per training episode.
- `train.n_support`. Number of support samples per training episode. Must be non-zero only if using prototypical loss.
- `train.n_query`. Number of query samples per class per training episode.
- `train.n_episodes`. Number of episodes per epoch.
- `train.loss`. Available: `triplet`, `prototypical`, `angproto`.
- `train.margin`. Loss margin parameter.


### Knowledge Distillation

Three complementary loss functions are used: **P2P** (point-to-point loss), **S2S** (structure-to-structure loss), and **KL** (KL divergence on prototype-anchored soft distributions).

#### Step 1 — Train the Teacher

Fine-tune an SSL backbone (Wav2Vec 2.0, HuBERT, ...) with LoRA adapters:

```bash
python KWSFSL/train_teacher.py \
  --data.cuda \
  --speech.dataset MSWC \
  --speech.task MSWC500U \
  --speech.default_datadir data/MSWC/audio/ \
  --speech.default_csvdir partitions/ \
  --speech.include_noise \
  --speech.noise_dir data/MSWC/noise/ \
  --teacher.model_name facebook/wav2vec2-base \
  --teacher.out_dim 276 \
  --teacher.lora_r 8 \
  --teacher.lora_alpha 16 \
  --train.epochs 40 \
  --train.n_way 40 \
  --train.n_support 0 \
  --train.n_query 10 \
  --train.n_episodes 100 \
  --train.loss triplet \
  --train.margin 0.5 \
  --log.exp_dir results/teacher_wav2vec/
```

Replace `facebook/wav2vec2-base` with `facebook/hubert-base-ls960` or other paths for other teachers. 


#### Step 2 — Knowledge Distillation

Train the student with a combination of triplet loss and distillation losses from one or more frozen teachers:

**Single teacher:**
```bash
python KWSFSL/train_distill.py \
  --data.cuda \
  --speech.dataset MSWC \
  --speech.task MSWC500U \
  --speech.default_datadir data/MSWC/audio/ \
  --speech.default_csvdir partitions/ \
  --speech.include_noise \
  --speech.noise_dir data/MSWC/noise/ \
  --model.model_name repr_conv \
  --model.encoding DSCNNL_LAYERNORM \
  --model.z_norm \
  --train.epochs 40 \
  --train.n_way 40 \
  --train.n_support 0 \
  --train.n_query 10 \
  --train.n_episodes 100 \
  --train.loss triplet \
  --train.margin 0.5 \
  --distill.teacher_paths results/teacher_wav2vec/best_model.pt \
  --distill.alpha_tl 1.0 \
  --distill.alpha_p2p 1.0 \
  --distill.alpha_s2s 1.0 \
  --distill.alpha_kl 1.0 \
  --log.exp_dir results/distill_student/
```

**Multiple teachers**:
```bash
python KWSFSL/train_distill.py \
  ... \
  --distill.teacher_paths results/teacher_wav2vec/best_model.pt \
                          results/teacher_hubert/best_model.pt \
  ...
```

#### Distillation Options
- `distill.teacher_paths`. Space-separated list of paths to trained teacher checkpoints.
- `distill.alpha_tl`. Weight for the triplet loss (set to 0 to disable).
- `distill.alpha_p2p`. Weight for the P2P embedding MSE loss.
- `distill.alpha_s2s`. Weight for the S2S similarity matrix loss.
- `distill.alpha_kl`. Weight for the KL prototype distribution loss.
- `distill.kl_temperature`. Temperature for the KL softmax.

Setting any alpha to 0 to skips that specific.


### Acknowledge

We acknowledge the following code repositories:
- https://github.com/ArchitParnami/Few-Shot-KWS
- https://github.com/roman-vygon/triplet_loss_kws
- https://github.com/clovaai/voxceleb_trainer
- https://github.com/BoLiu-SVCL/meta-open/
- https://github.com/tyler-hayes/Embedded-CL
- https://github.com/MrtnMndt/OpenVAE_ContinualLearning
- https://github.com/Codelegant92/STC-ProtoNet
- https://github.com/mrusci/ondevice-fewshot-kws

---

## Traditional Branch

### Installation

```bash
pip install -r requirements.txt
```

### Repository Overview

All the used scripts are included in the `kws_baseline/` folder:
* `config.py`: all hyperparameters (`AudioConfig`, `DataConfig`, `DTWConfig`, `HMMConfig`, `EvalConfig`)
* `data_loader.py`: GSC dataset download, few-shot split construction (enrollment / test / validation sets)
* `feature_extraction.py`: MFCC extraction with delta/delta-delta coefficients and CMVN normalization
* `dtw_baseline.py`: template-based DTW classifier with Sakoe-Chiba band
* `hmm_baseline.py`: left-to-right HMM and GMM-HMM with optional Universal Background Model (UBM); likelihood-based open-set scoring
* `evaluation.py`: accuracy, FRR, FAR, AUROC; ROC curves, confusion matrices, distance distributions
* `run_baselines.py`: main experiment script — runs multi-trial experiments, aggregates resultsm etc.

### Data Preparation

#### Google Speech Commands (GSC)

```bash
python kws_baseline/run_baselines.py --download
```

### Running Experiments

```bash
# Quick test (1 trial, 5-shot)
python kws_baseline/run_baselines.py --gsc_root ./data/SpeechCommands/speech_commands_v0.02 --k_shots 5 --n_trials 1

# Full experiment (5-shot and 10-shot, 5 trials)
python kws_baseline/run_baselines.py --gsc_root ./data/SpeechCommands/speech_commands_v0.02

# Custom run
python kws_baseline/run_baselines.py --k_shots 5 10 --n_trials 10 --output_dir ./results/full_run
```

Or use the convenience script:
```bash
bash run.sh
```