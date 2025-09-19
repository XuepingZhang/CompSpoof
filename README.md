# &#x20;Component-Level Audio Antispoofing (ICASSP 2026)

This repository contains the official implementation of our ICASSP 2025 paper:

**“**CompSpoof: A Dataset and Joint Learning Framework for Component-Level Audio Anti-spoofing Countermeasures**”**\
📄 [Paper on arXiv](https://arxiv.org/abs/xxxx.xxxxx) (coming soon)

***

## 🔊 CompSpoof Dataset

We introduce **CompSpoof**, a new dataset for component-level antispoofing, where either the speech component, the environmental component, or both can be spoofed.

*   📥 **Download link:** [CompSpoof dataset](https://huggingface.co/datasets/Sage99/CompSpoof)
*   📖 **Details & documentation:** [CompSpoof description](https://your-dataset-doc-link.com/)

***

## ⚙️ Setup

```
git clone https://github.com/XuepingZhang/CompSpoof.git
cd CompSpoof
conda create -n compspoof python=3.10
conda activate compspoof
pip install -r requirements.txt

```

***

## 🚀 Training & Evaluation

We provide scripts for model training and evaluation:

    # Training example
    python train.py

    # Evaluation example
    bash run_eval.sh
    python train.py --eval_path /path

You can modify the configs in conf.py to reproduce different experiments from the paper.

***

## 📊 Results on Dev Set

| Model      | Precision | Recall | F1     |
| :--------- | :-------- | :----- | :----- |
| Baseline   | 0.841     | 0.840  | 0.840  |
| Our method | 0.916     | 0.912  | 0.912  |

## 📊 Results on Eval Set

| Model      | Precision | Recall | F1    |
| :--------- | :-------- | :----- | :---- |
| Baseline   | 0.829     | 0.828  | 0.827 |
| Our method | 0.908     |  0.907 | 0.908 |

***

## ✨ Citation

If you find this work useful, please cite our paper:

```


```

