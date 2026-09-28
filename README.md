# Privacy–Hallucination Tradeoff


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

> **Work in progress.** The original research code has been reformatted slightly using AI agents to remove redundancy and make it more usable. 

## Citation

Our paper, [The Privacy-Hallucination Tradeoff in Differentially Private Language Models](https://arxiv.org/abs/2609.00492), was published at Findings of EMNLP 2026. If you use this code, please cite:

```bibtex
@misc{ramesh2026privacyhallucinationtradeoffdifferentiallyprivate,
      title={The Privacy-Hallucination Tradeoff in Differentially Private Language Models}, 
      author={Krithika Ramesh and Krishna Pillutla and Danish Pruthi and Anjalie Field},
      year={2026},
      eprint={2609.00492},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2609.00492}, 
}
```

## Contact

The reformatting of the code was done using AI agents, and while I have attempted to do sanity checks to ensure the faithfulness of the uploaded code, parts of it may have changed been changed in an unintended manner, In case of any problems, please raise an issue or contact me at public-krramesh[at]proton[dot]me.
