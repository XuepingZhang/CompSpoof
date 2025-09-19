import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取音频
wav_path = "/work/xz464/dataset/V3aaPyUdIyo_000254.wav"   # 换成你的音频文件路径
y, sr = librosa.load(wav_path, sr=None)

# 2. STFT
n_fft = 1024
hop_length = 256
D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

# 3. 可视化 (彩色)
plt.figure(figsize=(10, 4))
librosa.display.specshow(S_db, sr=sr, hop_length=hop_length,
                         x_axis='time', y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title("Spectrogram (STFT)")
plt.tight_layout()
plt.savefig("/work/xz464/composition_antispoofing_dkucc/feature/spec_env.png", dpi=300, bbox_inches='tight')
plt.show()
