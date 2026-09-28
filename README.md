# Privacy–Hallucination Tradeoff

> **Work in progress.** The original research code has been reformatted slightly to remove redundancy and make it more usable.

This code is derived from existing libraries: [synthtexteval](https://github.com/kr-ramesh/synthtexteval), [Opacus](https://opacus.ai/) and [dp-transformers](https://github.com/microsoft/dp-transformers)

## Contents

- `train/`: LoRA fine-tuning with and without differential privacy (DP-SGD with PLD noise calibration), and generation from trained models.
- `factuality_eval/`: FActScore-style factuality evaluation of generated text against a knowledge database.
- `datasets/`: Wikipedia datasets used for training and evaluation.

## Setup

```bash
conda create -n pht python=3.11 -y
conda activate pht
pip install -r requirements.txt
```


## Contact

While I've attempted to do sanity checks, in case of any issues, contact me at public-krramesh@proton.me.
