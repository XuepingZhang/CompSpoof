import os
import soundfile as sf
import numpy as np

# label_file = "/SMIIPdata2/zxp/composition_antispoofing/labels/full/dev_label1.txt"
# data_root = "/SMIIPdata2/zxp/composition_antispoofing/dataset"

label_file = '/work/xz464/labels/full/dev_label1.txt'
data_root = '/work/xz464/dataset'


os.makedirs(data_root, exist_ok=True)


import numpy as np
import soundfile as sf

def load_wav(path, target_sr=16000):
    wav, sr = sf.read(path)
    # 如果是多通道（stereo），取平均变成单通道
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    return wav.astype(np.float32), sr

def normalize_energy(wav):
    return wav / (np.sqrt(np.mean(wav ** 2)) + 1e-8)


def mix_wavs(speech, noise, snr_db=10):
    """
    混合说话人语音 (speech) 和背景声音 (noise)，背景音量按目标SNR缩放

    参数:
        speech: ndarray, 说话人语音
        noise: ndarray, 背景声音
        snr_db: float, 目标信噪比(dB)，越大表示语音越突出
    """
    # 对齐长度
    min_len = min(len(speech), len(noise))
    speech, noise = speech[:min_len], noise[:min_len]

    # 计算能量
    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)

    # 按目标 SNR 调整噪声
    snr_linear = 10 ** (snr_db / 10)
    target_noise_power = speech_power / snr_linear
    scale = np.sqrt(target_noise_power / (noise_power + 1e-8))
    noise = noise * scale

    # 相加混合
    mix = speech + noise

    # 防止 clipping
    mix = mix / (np.max(np.abs(mix)) + 1e-8)
    return mix


with open(label_file, "r") as f:
    lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue

        out_name, f1, f2 = parts[0], parts[1], parts[2]

        path1 = os.path.join(data_root, f1)
        path2 = os.path.join(data_root, f2)
        out_path = os.path.join(data_root, out_name)

        if not (os.path.exists(path1) and os.path.exists(path2)):
            print(f"跳过 {out_name}, 找不到文件 {f1} 或 {f2}")
            continue

        # 读取音频
        wav1, sr1 = load_wav(path1)
        wav2, sr2 = load_wav(path2)

        if sr1 != sr2:
            print(f"采样率不一致 {f1}({sr1}) vs {f2}({sr2}), 跳过 {out_name}")
            continue

        # 对齐长度
        min_len = min(len(wav1), len(wav2))
        wav1, wav2 = wav1[:min_len], wav2[:min_len]

        # 归一化能量
        wav1 = normalize_energy(wav1)
        wav2 = normalize_energy(wav2)

        # 混音
        mix = mix_wavs(wav1, wav2, snr_db=10)

        # 保存（覆盖已有文件）
        sf.write(out_path, mix, sr1)
        print(f"生成: {out_path}")
