# wujian@2018

import random
import torch as th
import numpy as np

from torch.utils.data.dataloader import default_collate
import torch.utils.data as dat
from .audio import WaveReader



def make_dataloader(train=True,
                    data_kwargs=None,
                    num_workers=4,
                    chunk_size=32000,
                    batch_size=16):
    mix_data = {}
    ref_data0 = {}
    ref_data1 = {}
    labels = {}

    label_map = {"spoof_spoof": [0,0,0], "bonafide_spoof": [1,0,0]
        , "spoof_bonafide": [0,1,0], "bonafide_bonafide": [1,1,0], "original": [1,1,1]}
    with open(data_kwargs, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 4:
                continue  # 跳过不完整的行
            a, b, c, d = parts[:4]
            mix_data[i] = a
            ref_data0[i] = b
            ref_data1[i] = c
            labels[i] = label_map[d]
    ref_data = [ref_data0, ref_data1]
    dataset = Dataset_(mix_data,ref_data,labels)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        train=train,
        num_workers=num_workers
    )


class Dataset_(object):
    """
    Per Utterance Loader
    """

    def __init__(self, mix_scp="", ref_scp=None, labels=None, sample_rate=16000):
        self.mix = WaveReader(mix_scp, sample_rate=sample_rate)
        self.ref = [WaveReader(ref, sample_rate=sample_rate) for ref in ref_scp]
        self.labels = labels

    def __len__(self):
        return len(self.mix)

    def __getitem__(self, index):
        key = self.mix.index_keys[index]

        # 获取 mix 音频
        mix = self.mix[key][0].astype(np.float32)
        path = self.mix[key][1]
        mix_len = len(mix)

        # 对 ref 做截取或补齐
        ref = []
        for reader in self.ref:
            r = reader[key][0].astype(np.float32)
            if len(r) >= mix_len:
                r = r[:mix_len]
            else:
                # 长度不足则补零
                r = np.pad(r, (0, mix_len - len(r)), mode="constant")
            ref.append(r)

        label = np.array(self.labels[index], dtype=np.float32)

        return {
            "mix": mix,
            "ref": ref,
            "label": label,
            "file":path
        }



class ChunkSplitter(object):
    """
    Split utterance into small chunks
    """
    def __init__(self, chunk_size, train=True, least=16000):
        self.chunk_size = chunk_size
        self.least = least
        self.train = train

    def _make_chunk(self, eg, s):
        """
        Make a chunk instance, which contains:
            "mix": ndarray,
            "ref": [ndarray...]
        """
        chunk = dict()
        chunk["mix"] = eg["mix"][s:s + self.chunk_size]
        chunk["ref"] = [ref[s:s + self.chunk_size] for ref in eg["ref"]]
        chunk["label"] = eg["label"]
        chunk["file"] = eg["file"]

        return chunk

    def split(self, eg):
        N = eg["mix"].size
        # too short, throw away
        if N < self.least:
            return []
        chunks = []
        # padding zeros
        if N < self.chunk_size:
            P = self.chunk_size - N
            chunk = dict()
            chunk["mix"] = np.pad(eg["mix"], (0, P), "constant")
            chunk["ref"] = [
                np.pad(ref, (0, P), "constant") for ref in eg["ref"]
            ]
            chunk["label"] = eg["label"]
            chunk["file"] = eg["file"]

            chunks.append(chunk)
        else:
            # random select start point for training
            s = random.randint(0, N % self.least) if self.train else 0
            while True:
                if s + self.chunk_size > N:
                    break
                chunk = self._make_chunk(eg, s)
                chunks.append(chunk)
                s += self.least
        return chunks


class DataLoader(object):
    """
    Online dataloader for chunk-level PIT
    """
    def __init__(self,
                 dataset,
                 num_workers=4,
                 chunk_size=32000,
                 batch_size=16,
                 train=True):
        self.batch_size = batch_size
        self.train = train
        self.splitter = ChunkSplitter(chunk_size,
                                      train=train,
                                      least=chunk_size // 2)
        # just return batch of egs, support multiple workers
        self.eg_loader = dat.DataLoader(dataset,
                                        batch_size=batch_size // 2,
                                        num_workers=num_workers,
                                        shuffle=train,
                                        collate_fn=self._collate)

    def _collate(self, batch):
        """
        Online split utterances
        """
        chunk = []
        for eg in batch:
            chunk += self.splitter.split(eg)
        return chunk

    def _merge(self, chunk_list):
        """
        Merge chunk list into mini-batch
        """
        N = len(chunk_list)
        if self.train:
            random.shuffle(chunk_list)
        blist = []
        for s in range(0, N - self.batch_size + 1, self.batch_size):
            batch = default_collate(chunk_list[s:s + self.batch_size])
            blist.append(batch)
        rn = N % self.batch_size
        return blist, chunk_list[-rn:] if rn else []

    def __iter__(self):
        chunk_list = []
        for chunks in self.eg_loader:
            chunk_list += chunks
            batch, chunk_list = self._merge(chunk_list)
            for obj in batch:
                yield obj