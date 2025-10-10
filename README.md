# &#x20;Component-Level Audio Antispoofing (ICASSP 2026)

This repository contains the official implementation of our ICASSP 2025 paper:

**“**CompSpoof: A Dataset and Joint Learning Framework for Component-Level Audio Anti-spoofing Countermeasures**”**\
📄 [Paper on arXiv](https://arxiv.org/abs/2509.15804)

***

## 🔊 CompSpoof Dataset

We introduce **CompSpoof**, a new dataset for component-level antispoofing, where either the speech component, the environmental component, or both can be spoofed.

*   📥 **Download link:** [CompSpoof dataset](https://huggingface.co/datasets/XuepingZhang/CompSpoof/)
*   📖 **Details & documentation:** [CompSpoof dataset description](https://xuepingzhang.github.io/CompSpoof-dataset/)

### 🎧 Audio Examples
Below are audio samples from the **CompSpoof** dataset. For each class, we provide the **mixed/original audio**, along with the **speech** and **environment** sources. 


#### Class 0 — Original 

**Label:** original 

**Description:** Original bona fide speech and corresponding environment audio without mixing

<table>
  <thead>
    <tr>
      <th>Original</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <audio controls>
          [bonafide_0_028.mp3](audio_demo/class0/bonafide_0_028.mp3)
        </audio>
      </td>
    </tr>
  </tbody>
</table>

#### Class 1 — Bona fide + Bona fide 

**Label:** bonafide_bonafide 

**Description:** Bona fide speech mixed with another bona fide environmental audio

<table>
  <thead>
    <tr>
      <th>Mixed</th>
      <th>Speech</th>
      <th>Environment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <audio controls>
          <source src="audio_demo/class1/bonafide_bonafide_0471.wav" type="audio/mpeg">
        </audio>
      </td>
      <td>
      <audio controls>
          <source src="audio_demo/class1/D_0002500345.flac" type="audio/mpeg">
        </audio>
      </td>
      <td>
      <audio controls>
          <source src="audio_demo/class1/-OQ3KFwzLCI_474.mp3" type="audio/mpeg">
        </audio>
      </td>
    </tr>
  </tbody>
</table>


#### Class 2 — Spoofed Speech + Bona fide Environment 

**Label:** spoof_bonafide 

**Description:** Spoof speech mixed with bona fide environmental audio
<table>
  <thead>
    <tr>
      <th>Mixed</th>
      <th>Speech</th>
      <th>Environment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <audio controls>
          <source src="audio_demo/class2/spoof_bonafide_0099.wav" type="audio/mpeg">
        </audio>
      </td>
      <td>
      <audio controls>
          <source src="audio_demo/class2/T_0000011037.flac" type="audio/mpeg">
        </audio>
      </td>
      <td>
      <audio controls>
          <source src="audio_demo/class2/-LGTb-xyjzA_11.mp3" type="audio/mpeg">
        </audio>
      </td>
    </tr>
  </tbody>
</table>


#### Class 3 — Bona fide Speech + Spoofed Environment 

**Label:** bonafide_spoof 

**Description:** Bona fide speech mixed with spoof environmental audio

<table>
  <thead>
    <tr>
      <th>Mixed</th>
      <th>Speech</th>
      <th>Environment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <audio controls>
          <source src="audio_demo/class3/bonafide_spoof_0248.wav" type="audio/mpeg">
        </audio>
      </td>
      <td>
      <audio controls>
          <source src="audio_demo/class3/D_0001820722.flac" type="audio/mpeg">
        </audio>
      </td>
      <td>
      <audio controls>
          <source src="audio_demo/class3/ViP3M-Hlm18_000030.wav" type="audio/mpeg">
        </audio>
      </td>
    </tr>
  </tbody>
</table>


#### Class 4 — Spoofed Speech + Spoofed Environment 

**Label:** spoof_spoof 

**Description:** Spoof speech mixed with spoof environmental audio

<table>
  <thead>
    <tr>
      <th>Mixed</th>
      <th>Speech</th>
      <th>Environment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <audio controls>
          <source src="audio_demo/class4/spoof_spoof_0439.wav" type="audio/mpeg">
        </audio>
      </td>
      <td>
      <audio controls>
          <source src="audio_demo/class4/T_0000141802.flac" type="audio/mpeg">
        </audio>
      </td>
      <td>
      <audio controls>
          <source src="audio_demo/class4/f_8Jnw9bU64_000008.wav" type="audio/mpeg">
        </audio>
      </td>
    </tr>
  </tbody>
</table>



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
@misc{zhang2025compspoofdatasetjointlearning,
      title={CompSpoof: A Dataset and Joint Learning Framework for Component-Level Audio Anti-spoofing Countermeasures}, 
      author={Xueping Zhang and Liwei Jin and Yechen Wang and Linxi Li and Ming Li},
      year={2025},
      eprint={2509.15804},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2509.15804}, 
}
```

