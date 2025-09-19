# wujian@2018
import logging
import os
import sys
import time

import numpy as np
import torch.optim as optim
from itertools import permutations
from collections import defaultdict
from collections import defaultdict, Counter
import random
import torchaudio
import os
import torch as th
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
from .metrics import calculate_EER_DCF_eval
from .utils import get_logger
import torch.nn as nn
import os
import random
import torch
import soundfile as sf
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.metrics import classification_report

def load_obj(obj, device):
    """
    Offload tensor object in obj to cuda device
    """

    def cuda(obj):
        return obj.to(device) if isinstance(obj, th.Tensor) else obj

    if isinstance(obj, dict):
        return {key: load_obj(obj[key], device) for key in obj}
    elif isinstance(obj, list):
        return [load_obj(val, device) for val in obj]
    else:
        return cuda(obj)


class SimpleTimer(object):
    """
    A simple timer
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.start = time.time()

    def elapsed(self):
        return (time.time() - self.start) / 60


class ProgressReporter(object):
    """
    A simple progress reporter
    """

    def __init__(self, logger, period=100):
        self.period = period
        self.logger = logger
        self.loss = []
        self.timer = SimpleTimer()

    def add(self, loss):
        self.loss.append(loss)
        N = len(self.loss)
        if not N % self.period:
            avg = sum(self.loss[-self.period:]) / self.period
            self.logger.info("Processed {:d} batches"
                             "(loss = {:+.2f})...".format(N, avg))

    def report(self, details=False):
        N = len(self.loss)
        if details:
            sstr = ",".join(map(lambda f: "{:.2f}".format(f), self.loss))
            self.logger.info("Loss on {:d} batches: {}".format(N, sstr))
        return {
            "loss": sum(self.loss) / N,
            "batches": N,
            "cost": self.timer.elapsed()
        }




class Trainer(object):
    def __init__(self,
                 nnet,
                 checkpoint="checkpoint",
                 optimizer="adam",
                 gpuid=0,
                 optimizer_kwargs=None,
                 clip_norm=None,
                 min_lr=0,
                 patience=0,
                 factor=0.5,
                 logging_period=100,
                 resume=None,
                 eval_path=None,
                 no_impr=5,
                 start_end=3):
        if not th.cuda.is_available():
            raise RuntimeError("CUDA device unavailable...exist")
        if not isinstance(gpuid, tuple):
            gpuid = (gpuid, )

        self.device = f"cuda:{gpuid[0]}"
        self.eval_path = eval_path

        self.gpuid = gpuid
        if checkpoint and not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        self.checkpoint = checkpoint
        self.logger = get_logger(
            os.path.join(checkpoint, "trainer.log"), file=True)

        self.clip_norm = clip_norm
        self.logging_period = logging_period
        self.cur_epoch = 0  # zero based
        self.no_impr = no_impr
        self.start_end = start_end
        if resume:
            if not os.path.exists(resume):
                raise FileNotFoundError(
                    "Could not find resume checkpoint: {}".format(resume))
            cpt = th.load(resume, map_location="cpu")
            self.cur_epoch = cpt["epoch"]
            self.logger.info("Resume from checkpoint {}: epoch {:d}".format(
                resume, self.cur_epoch))
            # load nnet
            nnet.load_state_dict(cpt["model_state_dict"])

            self.nnet = nnet.to(self.device)
            # self.nnet = nn.DataParallel(nnet, device_ids=self.gpuid).to(self.device)

            self.optimizer = self.make_optimizer(self.nnet)
            # self.create_optimizer(
            #     optimizer, optimizer_kwargs, state=cpt["optim_state_dict"])
        else:

            self.nnet = nnet.to(self.device)
            # self.nnet = nn.DataParallel(nnet, device_ids=self.gpuid).to(self.device)
            self.optimizer = self.make_optimizer(self.nnet)
            # self.optimizer = self.create_optimizer(optimizer, optimizer_kwargs)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=factor,
            patience=patience,
            min_lr=min_lr)
        self.num_params = sum(
            [param.nelement() for param in nnet.parameters()]) / 10.0**6

        # logging
        self.logger.info("Model summary:\n{}".format(nnet))
        self.logger.info("Loading model to GPUs:{}, #param: {:.2f}M".format(
            gpuid, self.num_params))
        if clip_norm:
            self.logger.info(
                "Gradient clipping by {}, default L2".format(clip_norm))

    def make_optimizer(self, model):
        optimizer = optim.Adam([
            # spar → lr=1e-3
            {"params": model.spar.parameters(), "lr": 1e-3},
            # 其他模块 → lr=1e-5
            # {"params": model.ssl_model.parameters(), "lr": 1e-5},
            {"params": model.aasist_all.parameters(), "lr": 1e-5},
            {"params": model.aasist_speech.parameters(), "lr": 1e-5},
            {"params": model.aasist_env.parameters(), "lr": 1e-5},
        ])
        return optimizer
    def save_checkpoint(self, best=True):
        cpt = {
            "epoch": self.cur_epoch,
            "model_state_dict": self.nnet.state_dict(),
            "optim_state_dict": self.optimizer.state_dict()
        }
        th.save(
            cpt,
            os.path.join(self.checkpoint,
                         "{0}.pt.tar".format("best" if best else "last")))

    def create_optimizer(self, optimizer, kwargs, state=None):
        supported_optimizer = {
            "sgd": th.optim.SGD,  # momentum, weight_decay, lr
            "rmsprop": th.optim.RMSprop,  # momentum, weight_decay, lr
            "adam": th.optim.Adam,  # weight_decay, lr
            "adadelta": th.optim.Adadelta,  # weight_decay, lr
            "adagrad": th.optim.Adagrad,  # lr, lr_decay, weight_decay
            "adamax": th.optim.Adamax  # lr, weight_decay
            # ...
        }
        if optimizer not in supported_optimizer:
            raise ValueError("Now only support optimizer {}".format(optimizer))
        opt = supported_optimizer[optimizer](self.nnet.parameters(), **kwargs)
        self.logger.info("Create optimizer {0}: {1}".format(optimizer, kwargs))
        if state is not None:
            opt.load_state_dict(state)
            self.logger.info("Load optimizer state dict from checkpoint")
        return opt

    def compute_loss(self, egs, end=False):
        raise NotImplementedError

    def metrics1(self, egs):
        # spks x n x S
        res = th.nn.parallel.data_parallel(self.nnet, egs, device_ids=self.gpuid)

        speech_, env_, res_speech_, res_env_, res_speech, res_env, res_all, h_all, h_speech_, h_env_, h_speech, h_env = res

        ests = (speech_, env_)
        # spks x n x S
        refs = egs["ref"]
        num_spks = len(refs)

        def sisnr_loss(permute):
            # for one permute
            return sum(
                [self.sisnr(ests[s], refs[t])
                 for s, t in enumerate(permute)]) / len(permute)

        # P x N
        N = egs["mix"].size(0)
        sisnr_mat = th.stack(
            [sisnr_loss(p) for p in permutations(range(num_spks))])
        max_perutt, _ = th.max(sisnr_mat, dim=0)
        # si-snr

        pit_sisnr = -th.sum(max_perutt) / N

        # ===== 分类损失 前面几个epoch不用分离得到的预测结果，后面几个才用=====


        # ===== 一致性评价 评估分离模型保留antispoofing特征的程度=====
        log_p_speech_ = F.softmax(res_speech_, dim=-1)
        p_speech = F.softmax(res_speech.detach(), dim=-1)
        L_cons_speech = F.cosine_similarity(log_p_speech_, p_speech, dim=-1).mean()

        log_p_env_ = F.softmax(res_env_, dim=-1)
        p_env = F.softmax(res_env.detach(), dim=-1)
        L_cons_env = F.cosine_similarity(log_p_env_, p_env, dim=-1).mean()

        L_cons = L_cons_speech + L_cons_env

        return (res_all, egs['label'].T[2]), \
                    (res_speech_, egs['label'].T[0]), \
                    (res_env_, egs['label'].T[1]), \
                    pit_sisnr, L_cons


    def only_eval(self, model_path, eval_loader):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Eval mode: model {model_path} not found")
        cpt = th.load(model_path, map_location="cpu")
        self.nnet.load_state_dict(cpt["model_state_dict"])
        self.logger.info(f"Loaded model from {model_path}, epoch {cpt['epoch']}")
        self.nnet.to(self.device)
        self.eval(eval_loader, mode="eval")
        self.eval_more(eval_loader, mode="eval")


    def train(self, data_loader, end):
        self.logger.info("Set train mode...")
        self.nnet.train()
        reporter = ProgressReporter(self.logger, period=self.logging_period)
        i = 0
        all_MSE = 0
        for egs in data_loader:
            # load to gpu
            egs = load_obj(egs, self.device)

            self.optimizer.zero_grad()
            loss,MSE = self.compute_loss(egs, end)
            loss.backward()
            if self.clip_norm:
                clip_grad_norm_(self.nnet.parameters(), self.clip_norm)
            self.optimizer.step()
            all_MSE = all_MSE + MSE
            reporter.add(loss.item())

            i = i+1
            # if i > 400:
            #     break
        return reporter.report(), all_MSE



    def save_random_example(self, egs, save_dir="./test", sr=16000):
        """
        egs: 一个 batch，包含 mix, ref0, ref1 (在GPU上)
        """
        os.makedirs(save_dir, exist_ok=True)

        # 随机选一个样本
        batch_size = egs["mix"].shape[0]

        idx = random.randint(0, batch_size - 1)

        # 取出 GPU tensor -> CPU numpy
        label = egs['label'][idx].detach().cpu()

        if label.tolist()[-1] != 1:
            mix = egs["mix"][idx].detach().cpu().numpy()
            ref0 = egs["ref"][0][idx].detach().cpu().numpy()
            ref1 = egs["ref"][1][idx].detach().cpu().numpy()

            # 保存 wav
            sf.write(os.path.join(save_dir, f"mix_{idx}_label{label.tolist()}.wav"), mix, sr)
            sf.write(os.path.join(save_dir, f"ref0_{idx}_label{label.tolist()}.wav"), ref0, sr)
            sf.write(os.path.join(save_dir, f"ref1_{idx}_label{label.tolist()}.wav"), ref1, sr)

    def train1(self, data_loader, end):
        self.logger.info("Set train mode...")
        self.nnet.train()
        reporter = ProgressReporter(self.logger, period=self.logging_period)
        i = 0
        all_MSE = 0

        for egs in data_loader:
            # load to gpu
            egs = load_obj(egs, self.device)

            self.optimizer.zero_grad()
            loss, MSE = self.compute_loss(egs, end)
            loss.backward()
            if self.clip_norm:
                clip_grad_norm_(self.nnet.parameters(), self.clip_norm)
            self.optimizer.step()
            all_MSE += MSE
            reporter.add(loss.item())

            # 每个 batch 随机保存一个 mix/ref0/ref1
            self.save_random_example(egs, save_dir="./test")

            i += 1

        return reporter.report(), all_MSE

    def eval(self, data_loader, mode):
        self.logger.info(f"Set {mode} mode...")
        self.nnet.eval()

        label_map = {
            (0, 0, 1): 0,
            (0, 1, 1): 0,
            (1, 0, 1): 0,
            (1, 1, 1): 0,
            (1, 1, 0): 1,
            (0, 1, 0): 2,
            (1, 0, 0): 3,
            (0, 0, 0): 4,
        }

        all_labels, all_preds = [], []
        file2labels, file2preds = defaultdict(list), defaultdict(list)
        save_idx = 0  # 保存音频的编号
        out_dir = "./test/"
        os.makedirs(out_dir, exist_ok=True)

        with th.no_grad():
            for egs in data_loader:
                egs = load_obj(egs, self.device)

                labels = egs["label"].cpu().numpy().tolist()
                true_labels = [label_map[tuple(lbl)] for lbl in labels]

                res = th.nn.parallel.data_parallel(self.nnet, egs, device_ids=self.gpuid)
                speech_, env_, res_speech_, res_env_, res_speech, res_env, res_all, h_all, h_speech_, h_env_, h_speech, h_env = res

                res_speech_ = res_speech_.argmax(dim=1).cpu().numpy()
                res_env_ = res_env_.argmax(dim=1).cpu().numpy()
                res_all = res_all.argmax(dim=1).cpu().numpy()

                pred_labels = [
                    label_map[(int(s), int(e), int(a))]
                    for s, e, a in zip(res_speech_, res_env_, res_all)
                ]

                all_labels.extend(true_labels)
                all_preds.extend(pred_labels)

                # 如果是 eval 模式，收集文件级别信息

                file_names = egs["file"]  # 假设是 list[str]
                for f, t, p in zip(file_names, true_labels, pred_labels):
                    file2labels[f].append(t)
                    file2preds[f].append(p)

                # ====== 新增：每个 batch 保存一组音频 ======
                idx_candidates = [i for i, t in enumerate(true_labels) if t != 0]
                if len(idx_candidates) > 0:
                    idx = random.choice(idx_candidates)

                    # 取出对应样本，转 CPU
                    speech_sample = speech_[idx].detach().cpu()
                    env_sample = env_[idx].detach().cpu()
                    ref0_sample = egs['ref'][0][idx].detach().cpu()
                    ref1_sample = egs['ref'][1][idx].detach().cpu()
                    mix_sample = egs['mix'][idx].detach().cpu()

                    # 保存音频 (默认16kHz)
                    torchaudio.save(os.path.join(out_dir, f"{mode}_mix_{save_idx}_label{true_labels[save_idx]}.wav"),
                                    mix_sample.unsqueeze(0), 16000)
                    torchaudio.save(os.path.join(out_dir, f"{mode}_speech_{save_idx}_label{true_labels[save_idx]}.wav"),
                                    speech_sample.unsqueeze(0), 16000)
                    torchaudio.save(os.path.join(out_dir, f"{mode}_env_{save_idx}_label{true_labels[save_idx]}.wav"),
                                    env_sample.unsqueeze(0), 16000)
                    torchaudio.save(os.path.join(out_dir, f"{mode}_speech_{save_idx}_tar_label{true_labels[save_idx]}.wav"),
                                    ref0_sample.unsqueeze(0), 16000)
                    torchaudio.save(os.path.join(out_dir, f"{mode}_env_{save_idx}_tar_label{true_labels[save_idx]}.wav"),
                                    ref1_sample.unsqueeze(0), 16000)

                    save_idx += 1
                # ====== 新增结束 ======

        # macro 平均
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average="macro")
        f1 = f1_score(all_labels, all_preds, average="macro")

        recall_per_class = recall_score(all_labels, all_preds, average=None)
        f1_per_class = f1_score(all_labels, all_preds, average=None)

        import numpy as np
        all_labels_arr = np.array(all_labels)
        all_preds_arr = np.array(all_preds)

        precision_per_class = []
        for cls in np.unique(all_labels_arr):
            pred_mask = (all_preds_arr == cls)
            if pred_mask.sum() == 0:
                precision_cls = 0.0
            else:
                precision_cls = (all_labels_arr[pred_mask] == cls).mean()
            precision_per_class.append(precision_cls)

        self.logger.info(f"{mode} Results - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
            self.logger.info(f"Class {i}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}")



        # 文件级别指标
        file_labels, file_preds = [], []
        for f in file2labels.keys():
            # 真实标签（多数情况下应该一致，取第一个即可）
            true_label = file2labels[f][0]
            # 投票决定预测标签
            pred_counter = Counter(file2preds[f])
            pred_label = pred_counter.most_common(1)[0][0]

            file_labels.append(true_label)
            file_preds.append(pred_label)

        # 宏平均
        file_acc = accuracy_score(file_labels, file_preds)
        file_precision = precision_score(file_labels, file_preds, average="macro")
        file_recall = recall_score(file_labels, file_preds, average="macro")
        file_f1 = f1_score(file_labels, file_preds, average="macro")

        # 每类指标
        report = classification_report(file_labels, file_preds, digits=4)

        self.logger.info("\n========== File-level Evaluation ==========")
        self.logger.info(f"Overall Acc: {file_acc:.4f}")
        self.logger.info(f"Macro Precision: {file_precision:.4f}, "
                         f"Macro Recall: {file_recall:.4f}, "
                         f"Macro F1: {file_f1:.4f}")
        self.logger.info("\nPer-class Results:\n" + report)
        self.logger.info("===========================================")

        return file_precision, file_recall, file_f1



    def eval_more(self, data_loader, mode):
        self.logger.info(f"Set {mode} mode...")
        self.nnet.eval()

        # 收集结果
        all_metrics = {
            "speech_sep": {"labels": [], "preds": []},
            "speech_ori": {"labels": [], "preds": []},
            "env_sep": {"labels": [], "preds": []},
            "env_ori": {"labels": [], "preds": []},
            "mix": {"labels": [], "preds": []},
        }

        # 隐藏层特征收集
        feats = {
            "speech_sep": {"x": [], "y": []},
            "speech_ori": {"x": [], "y": []},
            "env_sep": {"x": [], "y": []},
            "env_ori": {"x": [], "y": []},
            "mix": {"x": [], "y": []},
        }

        with th.no_grad():
            for egs in data_loader:
                egs = load_obj(egs, self.device)

                # 标签
                lbl_speech = egs["label"][:, 0].cpu().numpy()
                lbl_env = egs["label"][:, 1].cpu().numpy()
                lbl_mix = egs["label"][:, 2].cpu().numpy()

                # 模型前向
                res = th.nn.parallel.data_parallel(self.nnet, egs, device_ids=self.gpuid)
                (speech_, env_, res_speech_, res_env_, res_speech,
                 res_env, res_all, h_all, h_speech_, h_env_,
                 h_speech, h_env) = res

                # 预测
                pred_speech_ = res_speech_.argmax(dim=1).cpu().numpy()
                pred_speech = res_speech.argmax(dim=1).cpu().numpy()
                pred_env_ = res_env_.argmax(dim=1).cpu().numpy()
                pred_env = res_env.argmax(dim=1).cpu().numpy()
                pred_mix = res_all.argmax(dim=1).cpu().numpy()

                # 累积标签+预测
                all_metrics["speech_sep"]["labels"].extend(lbl_speech)
                all_metrics["speech_sep"]["preds"].extend(pred_speech_)
                all_metrics["speech_ori"]["labels"].extend(lbl_speech)
                all_metrics["speech_ori"]["preds"].extend(pred_speech)
                all_metrics["env_sep"]["labels"].extend(lbl_env)
                all_metrics["env_sep"]["preds"].extend(pred_env_)
                all_metrics["env_ori"]["labels"].extend(lbl_env)
                all_metrics["env_ori"]["preds"].extend(pred_env)
                all_metrics["mix"]["labels"].extend(lbl_mix)
                all_metrics["mix"]["preds"].extend(pred_mix)

                # ===== 保存隐藏层特征（只存二分类 spoof/bonafide） =====
                feats["speech_sep"]["x"].append(h_speech_.cpu().numpy())
                feats["speech_sep"]["y"].append(lbl_speech)
                feats["speech_ori"]["x"].append(h_speech.cpu().numpy())
                feats["speech_ori"]["y"].append(lbl_speech)
                feats["env_sep"]["x"].append(h_env_.cpu().numpy())
                feats["env_sep"]["y"].append(lbl_env)
                feats["env_ori"]["x"].append(h_env.cpu().numpy())
                feats["env_ori"]["y"].append(lbl_env)
                feats["mix"]["x"].append(h_all.cpu().numpy())
                feats["mix"]["y"].append(lbl_mix)

        # ====== 计算并打印指标 ======
        for name, d in all_metrics.items():
            labels, preds = np.array(d["labels"]), np.array(d["preds"])
            acc = accuracy_score(labels, preds)
            pre = precision_score(labels, preds, average="macro", zero_division=0)
            rec = recall_score(labels, preds, average="macro", zero_division=0)
            f1 = f1_score(labels, preds, average="macro", zero_division=0)
            self.logger.info(f"[{mode}] {name} - Acc={acc:.4f}, Pre={pre:.4f}, Rec={rec:.4f}, F1={f1:.4f}")

        # ====== 保存隐藏层特征 ======
        save_dir = os.path.join(self.checkpoint, 'feature')
        os.makedirs(save_dir, exist_ok=True)
        for name, d in feats.items():
            x = np.concatenate(d["x"], axis=0)
            y = np.concatenate(d["y"], axis=0)
            np.save(os.path.join(save_dir, f"{name}_x.npy"), x)
            np.save(os.path.join(save_dir, f"{name}_y.npy"), y)

        return

    def run(self, train_loader, dev_loader,eval_loader,split_epoch = None, num_epochs=50):
        # check if save is OK
        if self.eval_path is not None:
            self.only_eval(self.eval_path, eval_loader)
            return
        no_impr_MSE = 0
        no_impr = 0
        best_MSE = 10000000.
        best_dev_f1 = 0.
        # make sure not inf
        self.scheduler.best = best_dev_f1
        while self.cur_epoch < num_epochs:
            self.cur_epoch += 1
            cur_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info( "Loss(time/N, lr={:.3e}) - Epoch {:2d}:".format(
                    cur_lr, self.cur_epoch))

            tr,MSE = self.train(train_loader,self.cur_epoch >= self.start_end)
            self.logger.info("train = {:+.4f}({:.2f}m/{:d})".format(
                tr["loss"], tr["cost"], tr["batches"]))



            # schedule here

            # flush scheduler info
            sys.stdout.flush()

            dev_res = self.eval(dev_loader,mode='dev')
            self.scheduler.step(dev_res[2])
            if dev_res[2] < best_dev_f1:
                no_impr += 1
                self.logger.info("| no impr, best = {:.4f}".format(
                    best_dev_f1))
            else:
                best_dev_f1 = dev_res[2]
                no_impr = 0
                self.save_checkpoint(best=True)

            self.eval(eval_loader,mode='eval')
            self.eval_more(eval_loader,mode='eval')


            # save last checkpoint
            self.save_checkpoint(best=False)
            if no_impr == self.no_impr:
                self.logger.info(
                    "Stop training cause no impr for {:d} epochs".format(
                        no_impr))
                break
        self.logger.info("Training for {:d}/{:d} epoches done!".format(
            self.cur_epoch, num_epochs))


class SiSnrTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super(SiSnrTrainer, self).__init__(*args, **kwargs)

        self.criterion = nn.MSELoss()
    def sisnr(self, x, s, eps=1e-8):
        """
        Arguments:
        x: separated signal, N x S tensor
        s: reference signal, N x S tensor
        Return:
        sisnr: N tensor
        """

        def l2norm(mat, keepdim=False):
            return th.norm(mat, dim=-1, keepdim=keepdim)

        if x.shape != s.shape:
            raise RuntimeError(
                "Dimention mismatch when calculate si-snr, {} vs {}".format(
                    x.shape, s.shape))
        x_zm = x - th.mean(x, dim=-1, keepdim=True)
        s_zm = s - th.mean(s, dim=-1, keepdim=True)
        t = th.sum(
            x_zm * s_zm, dim=-1,
            keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
        return 20 * th.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

    def compute_loss(self, egs, end=False):
        # spks x n x S
        res = th.nn.parallel.data_parallel(self.nnet, egs, device_ids=self.gpuid)

        speech_, env_, res_speech_, res_env_, res_speech, res_env, res_all , h_all, h_speech_, h_env_, h_speech, h_env = res
        label_speech = egs['label'].T[0]
        label_env = egs['label'].T[1]
        label_all = egs['label'].T[2]

        # 整体分类损失 (总是计算)
        weights = th.tensor([0.2, 0.8], device=res_all.device, dtype=th.float32)
        L_cls_all = F.cross_entropy(res_all, label_all.long(), weight=weights)
        # mask: 只对 label_all == 0 的样本计算分离相关 loss
        mask = (label_all == 0)
        mask_count = mask.sum().item()  # 有多少个样本参与

        # ===== 分离损失 (MSER) =====

        MSE = self.criterion(speech_,egs['ref'][0]) + self.criterion(env_,egs['ref'][1])


        # ===== 分类损失 =====
        if end:
            L_cls_speech_, L_cls_env_ = 0.0, 0.0
            L_cons = 0.0
            if mask_count > 0:
                L_cls_speech_ = F.cross_entropy(res_speech_[mask], label_speech[mask].long())
                L_cls_env_ = F.cross_entropy(res_env_[mask], label_env[mask].long())

                # 一致性损失
                log_p_speech_ = F.log_softmax(res_speech_[mask], dim=-1)
                p_speech = F.softmax(res_speech.detach()[mask], dim=-1)
                L_cons_speech = F.kl_div(log_p_speech_, p_speech, reduction='batchmean')

                log_p_env_ = F.log_softmax(res_env_[mask], dim=-1)
                p_env = F.softmax(res_env.detach()[mask], dim=-1)
                L_cons_env = F.kl_div(log_p_env_, p_env, reduction='batchmean')

                L_cons = L_cons_speech + L_cons_env

            return L_cls_all + 10*MSE + L_cls_speech_ + L_cls_env_ + L_cons, 10*MSE

        else:
            L_cls_speech, L_cls_env = 0.0, 0.0
            if mask_count > 0:
                L_cls_speech = F.cross_entropy(res_speech[mask], label_speech[mask].long())
                L_cls_env = F.cross_entropy(res_env[mask], label_env[mask].long())

            return L_cls_all + 10*MSE + L_cls_speech + L_cls_env, 10*MSE
