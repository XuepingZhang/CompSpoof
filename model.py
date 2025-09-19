import logging
import random
from typing import Union
import torch as th
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from Unet.Unet_mask import UNetSTFTComplexRefine
from conf import aasist_conf,chunk_size
from waveunet import Waveunet

from XLSR2_AASIST import XLSR2_AASIST, SSLModel


class Model(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        # self.spar = Waveunet()
        self.spar = UNetSTFTComplexRefine()
        # self.ssl_model = SSLModel()  # 用于提取 SSL embedding (xlsr/wav2vec)

        self.aasist_all = XLSR2_AASIST(aasist_conf)
        self.aasist_speech = XLSR2_AASIST(aasist_conf)
        self.aasist_env = XLSR2_AASIST(aasist_conf)

        # EMA 更新速率（可调）
        # self.proto_momentum = 0.95

        # self.speech_proto = nn.Parameter(torch.zeros(1, 1024))

    def extract_embedding(self, waveform):
        """从波形提取 embedding（最后一层 -> mean pooling）"""
        with torch.no_grad():
            feat = self.ssl_model.extract_feat(waveform.squeeze(-1))  # [B, T, D]
            emb = feat.mean(dim=1)  # -> [B, D]
        return emb

    def update_prototypes(self, speech_emb):
        speech_mean = speech_emb.mean(dim=0, keepdim=True).detach()
        with torch.no_grad():
            if torch.count_nonzero(self.speech_proto) == 0:
                self.speech_proto.copy_(speech_mean)  # <- in-place 更新
            else:
                self.speech_proto.mul_(self.proto_momentum).add_(speech_mean * (1 - self.proto_momentum))

    def resort(self, speech_proto, waveform1, waveform2):
        """推理时根据 proto 重新排序"""
        emb1 = self.extract_embedding(waveform1)
        emb2 = self.extract_embedding(waveform2)

        sim1_speech = F.cosine_similarity(emb1, speech_proto, dim=-1)

        sim2_speech = F.cosine_similarity(emb2, speech_proto, dim=-1)


        # 决策：哪个更像 speech
        return self.reorder_waveforms(sim1_speech, sim2_speech, waveform1, waveform2)


    def reorder_waveforms(self, sim1_speech, sim2_speech, waveform1, waveform2):
        """
        sim1_speech, sim2_speech: [B] tensor
        waveform1, waveform2: [B, T] tensor
        返回: new_waveform1, new_waveform2
        """
        # mask: True 表示 sim1 > sim2，False 表示 sim2 >= sim1
        mask = sim1_speech > sim2_speech  # [B]

        # 初始化输出
        new_waveform1 = th.zeros_like(waveform1)
        new_waveform2 = th.zeros_like(waveform2)

        # 按 mask 选择
        new_waveform1[mask] = waveform1[mask]
        new_waveform2[mask] = waveform2[mask]

        new_waveform1[~mask] = waveform2[~mask]
        new_waveform2[~mask] = waveform1[~mask]

        return new_waveform1, new_waveform2

    def forward(self, egs):
        """egs 包含 mix, ref=[speech, env]"""
        h_all, res_all = self.aasist_all(egs['mix'])
        speech_, env_ = self.spar(egs['mix'])  # ConvTasNet 输出两个波形
        # speech_, env_ = res['speech'].squeeze()[:, :chunk_size] , res['env'].squeeze()[:, :chunk_size]

        # if self.training:
        #     # 从 GT ref 提取 embedding 来更新 prototype
        #     speech_emb = self.extract_embedding(egs["ref"][0])
        #     self.update_prototypes(speech_emb)
        #
        # else:
        #     # logging.info(self.speech_proto)
        #     # eval 模式，重新排序
        #     speech_, env_ = self.resort(self.speech_proto,speech_, env_)

        # 下游鉴别器

        h_speech_, res_speech_ = self.aasist_speech(speech_)
        h_env_, res_env_ = self.aasist_env(env_)
        h_speech, res_speech = self.aasist_speech(egs["ref"][0])
        h_env, res_env = self.aasist_env(egs["ref"][1])

        return speech_, env_\
            , res_speech_, res_env_, res_speech, res_env, res_all\
            , h_all, h_speech_, h_env_, h_speech, h_env


