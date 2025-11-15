# private_hybridNet_emg2qwerty

This repository contains the code used for the experiments presented in the paper:

> **“Hybrid, two-stage neural networks optimize the accuracy vs. computational cost trade-off in sEMG decoding.”**

---

## Overview

This repository includes tools to:

- Evaluate the proposed neural network models on the **emg2qwerty** dataset (both **user-specific** and **zero-shot** conditions), reproducing the key results of the paper;
- Retrain the proposed models discussed in the paper.

The experiments rely on the publicly available emg2qwerty dataset, which provides surface EMG recordings paired with text-entry labels. 
More information about the dataset structure and acquisition protocol can be found in the official repository: https://github.com/facebookresearch/emg2qwerty

---

## Setup

Clone the repository:

```bash
git clone https://github.com/andreorto98/private_hybridNet_emg2qwerty.git
```

Download, extract, and symlink the **emg2qwerty** dataset:

```bash
cd ~
wget https://fb-ctrl-oss.s3.amazonaws.com/emg2qwerty/emg2qwerty-data-2021-08.tar.gz
tar -xvzf emg2qwerty-data-2021-08.tar.gz
ln -s ~/emg2qwerty-data-2021-08 ~/emg2qwerty/data
```

> For more instructions on dataset usage and train/validation/test split generation, see the original repo:  
> https://github.com/facebookresearch/emg2qwerty

Install required Python packages:

```bash
# Recommended: create and activate a virtual environment first; we recommend using python3.10

virtualenv <env_name> --python=<path-to-python3.10>
source <env_name>/bin/activate

pip install -r requirements.txt

# When finished
deactivate
```

To properly set up the **EvNN: Event-based Neural Networks** package, please refer to:  
https://github.com/Efficient-Scalable-Machine-Learning/EvNN

---

## Usage

All experiments are managed via the provided python script:

```
run_model.py
```

### Inference

To run inference on a specific test user or in zero-shot conditions:

```bash
python3 run_model.py --user <user_id> --model <model_id> --root_dataset <path_to_dataset>
```

- `<user_id>`: one of the 8 test users (e.g., `user0`) or `generic` for **zero-shot** evaluation
- `<model_id>` options:
  - `S4_large`
  - `TC_S4_medium`
  - `TC_S4_small`  

Please refere to Table 1 in the manuscript for major details on these models.

Pretrained model checkpoints are provided in the `trained_checkpoints` directory.

### Training

To retrain a model, please run the script with the extra argument **--train**:

```bash
python3 run_model.py --user <user_id> --model <model_id> --root_dataset <path_to_dataset> --train
```

Please make sure to set up **Weights & Biases (wandb)** before running training, as it is used for experiment logging and metric visualization. You can enable it by creating a free account at https://wandb.ai, then logging in from the command line:

```bash
wandb login
```

---



