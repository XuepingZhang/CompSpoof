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
from dasheng import dasheng_base
from XLSR2_AASIST import XLSR2_AASIST, SSLModel
from dasheng_aasist import Dasheng_AASIST

class Model(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.spar = UNetSTFTComplexRefine()


        self.aasist_all = XLSR2_AASIST(aasist_conf)
        self.aasist_speech = XLSR2_AASIST(aasist_conf)
        self.aasist_env = Dasheng_AASIST(aasist_conf)


    def forward(self, egs):
        """egs 包含 mix, ref=[speech, env]"""
        h_all, res_all = self.aasist_all(egs['mix'])
        speech_, env_ = self.spar(egs['mix'])  # ConvTasNet 输出两个波形
        # speech_, env_ = res['speech'].squeeze()[:, :chunk_size] , res['env'].squeeze()[:, :chunk_size]



        # 下游鉴别器

        h_speech_, res_speech_ = self.aasist_speech(speech_)
        h_env_, res_env_ = self.aasist_env(env_)
        h_speech, res_speech = self.aasist_speech(egs["ref"][0])
        h_env, res_env = self.aasist_env(egs["ref"][1])

        return speech_, env_\
            , res_speech_, res_env_, res_speech, res_env, res_all\
            , h_all, h_speech_, h_env_, h_speech, h_env


