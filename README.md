# &#x20;Component-Level Audio Antispoofing (ICASSP 2026, under review)

This repository contains the official implementation of our ICASSP 2026 paper (under review):

**“**CompSpoof: A Dataset and Joint Learning Framework for Component-Level Audio Anti-spoofing Countermeasures**”**\
📄 [Paper on arXiv](https://arxiv.org/abs/2509.15804)

***


## 📢 NEWS!!
We expanded CompSpoof dataset to CompSpoofV2, which significantly expands the diversity of attack sources, environmental sounds, and mixing strategies. ✨ In addition, newly generated audio samples are distributed across the test set and are specifically designed to serve as detection data under unseen conditions.

🤗 CompSpoofV2 Details & Download Link: https://xuepingzhang.github.io/CompSpoof-V2-Dataset/

Building upon CompSpoofV2 dataset and separation-enhanced joint learning framework, we lunched the **ICME 2026 Environment-Aware Speech and Sound Deepfake Detection Challenge (ESDD2)**. We warmly invite researchers from both academia and industry to participate in this challenge, exploring robust and effective solutions for these critical deepfake detection tasks.

🖥️ ESDD2 website: https://sites.google.com/view/esdd-challenge/esdd-challenges/esdd-2/description

---

## 🔊 CompSpoof Dataset

We introduce **CompSpoof**, a new dataset for component-level antispoofing, where either the speech component, the environmental component, or both can be spoofed.

*   📥 **Download link:** [CompSpoof dataset](https://huggingface.co/datasets/XuepingZhang/CompSpoof/)
*   📖 **Details & documentation & Audio Examples:** [CompSpoof dataset description](https://xuepingzhang.github.io/CompSpoof-dataset/)

***


## ⚙️ Setup

```
git clone https://github.com/XuepingZhang/CompSpoof.git
cd CompSpoof
conda create -n compspoof python=3.10
conda activate compspoof
git clone https://github.com/facebookresearch/fairseq.git fairseq_dir
cd fairseq_dir
git checkout a54021305d6b3c
pip install --editable ./
pip install -r requirements.txt
```

**Pretrained XLSR**

The pretrained model XLSR can be found at this [link](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt).

***

## 🚀 Training & Evaluation

We provide scripts for model training and evaluation:

    # Training example
    python train.py

    # Evaluation example
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
@inproceedings{zhang2025compspoofdatasetjointlearning,
      title={CompSpoof: A Dataset and Joint Learning Framework for Component-Level Audio Anti-spoofing Countermeasures}, 
      author={Xueping Zhang and Liwei Jin and Yechen Wang and Linxi Li and Ming Li},
      year={2026},
      booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
}
```



## 🔏 License 

 The code is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/deed.en) license.

---

## ✉️  Contact Information

For questions, issues, or collaboration inquiries, please contact:
* Email: [xueping.zhang@dukekunshan.edu.cn](xueping.zhang@dukekunshan.edu.cn)