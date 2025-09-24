import os

import numpy as np
import soundfile as sf

label_file = '/work/xz464/labels/full/dev_label1.txt'
data_root = '/work/xz464/dataset'
os.makedirs(data_root, exist_ok=True)


def load_wav(path, target_sr=16000):
    wav, sr = sf.read(path)

    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    return wav.astype(np.float32), sr

def normalize_energy(wav):
    return wav / (np.sqrt(np.mean(wav ** 2)) + 1e-8)


def mix_wavs(speech, noise, snr_db=10):

    min_len = min(len(speech), len(noise))
    speech, noise = speech[:min_len], noise[:min_len]

    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)

    snr_linear = 10 ** (snr_db / 10)
    target_noise_power = speech_power / snr_linear
    scale = np.sqrt(target_noise_power / (noise_power + 1e-8))
    noise = noise * scale

    mix = speech + noise

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
            print(f"Skip {out_name}, can not find {f1} or {f2}")
            continue

        wav1, sr1 = load_wav(path1)
        wav2, sr2 = load_wav(path2)

        if sr1 != sr2:
            print(f"Inconsistent sampling rates {f1}({sr1}) vs {f2}({sr2}), skip {out_name}")
            continue

        min_len = min(len(wav1), len(wav2))
        wav1, wav2 = wav1[:min_len], wav2[:min_len]

        wav1 = normalize_energy(wav1)
        wav2 = normalize_energy(wav2)

        mix = mix_wavs(wav1, wav2, snr_db=10)

        sf.write(out_path, mix, sr1)
