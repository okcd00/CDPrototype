"""
@Time   :   2021-01-21 10:57:33
@File   :   model_example.py
@Author :   okcd00
@Email  :   okcd00@qq.com
"""

import numpy as np
from pprint import pprint
from collections import defaultdict

import torch
from transformers import BertTokenizer
from torch.utils.tensorboard import SummaryWriter

from utils import flatten
from utils.text_utils import split_2_short_text
from utils.evaluations import report_prf
from .bases import BaseTrainingEngine


class CscTrainingModel(BaseTrainingEngine):
    """
    用于CSC的BaseModel, 定义了训练及预测步骤
    """

    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        # loss weight for cor & det
        self.w = cfg.MODEL.HYPER_PARAMS[0]
        self.recorder = defaultdict(int)
        self.tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_CKPT)
        self.tb_writer = SummaryWriter(cfg.OUTPUT_DIR + '/tensorboard_logs/')
        # self.tokenizer._add_tokens(['“', '”'])

        # threshold for prediction judgment
        self.judge_line = 0.5
        self.has_explict_detector = True
        self.show_result_steps = int(2e4)  # 20k steps per hour

        # record lists
        self.train_loss_epoch = []
        self.train_loss_window = []
        self.count_matrix = None
        self.reset_matrix()        
        self.board_writer = None

    @staticmethod
    def pt2list(_tensor):
        return _tensor.cpu().numpy().tolist()

    def logging_all(self, item, name):
        self._logger.info(f'{name}: {item}')
        self.log(f'{name}', item)
        self.tb_writer.add_scalar(f'{name}', item, self.recorder[f'{name}'])
        # auto indexing
        self.recorder[f'{name}'] += 1

    def update_mean_value(self, case, val, cnt=1):
        if len(case) != 2:
            val_mean, val_count = 0., 0.
        else:    
            val_mean, val_count = case[0], case[1]
        val_mean = (val_mean * val_count + val * cnt) / (val_count + cnt)
        val_count += cnt
        return [val_mean, val_count]

    def update_matrix(self, details):
        for key in self.count_matrix:
            if key in details:
                upd = details[key]
                for idx in range(3):
                    self.count_matrix[key][idx] += upd[idx]

    def reset_matrix(self):
        count_matrix = {}
        for key in ['det_char', 'cor_char', 'det_sent', 'cor_sent']:
            count_matrix[f"{key}_counts"] = [0, 0, 0]  # [TP, FP, FN]
        self.count_matrix = count_matrix

    def print_prf_for_fusion_matrix(self, n_step):
        record = (0., 0., 0.)
        for key in self.count_matrix:
            # TP, FP, FN = self.count_matrix[key]
            precision, recall, f1_score = report_prf(
                *self.count_matrix[key], logger=self._logger, 
                phase=f"{key.replace('_counts', '')} at {n_step}-th steps")
            if key.startswith('cor_sent'):
                record = (precision, recall, f1_score)
        return record

    def get_encoded_texts(self, texts):
        if texts is None:
            return None
        encoded_texts = self.tokenizer(texts, padding=True, return_tensors='pt').to(self._device)
        return encoded_texts

    def get_results_from_outputs(self, outputs, batch):
        results = []
        # do something about outputs(pred) and the batch(truth)
        # results for calculating PRF
        return results

    def show_train_windows(self, n_step=0):
        _, _, _f = self.print_prf_for_fusion_matrix(n_step=n_step)
        self.reset_matrix()  # reset for next round
        avg_trn_loss = 0.
        if self.train_loss_window:
            avg_trn_loss = np.mean(self.train_loss_window)
        
        # logger info
        self.logging_all(avg_trn_loss, name='average_train_loss')
        self.logging_all(_f, name='average_train_f1')
        self.train_loss_window = []

    def training_step(self, batch, batch_idx):
        
        outputs = self.forward(*batch)
        loss = 0.  # get loss from outputs
        
        if self.show_result_steps > 0:
            # record results
            results = self.get_results_from_outputs(outputs, batch)
            
            _loss = loss.cpu().item()
            self.train_loss_window.append(_loss)
            # curr_epoch = self.current_epoch
            # curr_step = self.global_step
            self.tb_writer.add_scalar('train_loss', _loss, self.global_step)

            # show results for every k steps
            if batch_idx > 0 and self.show_result_steps != -1:
                if batch_idx % self.show_result_steps == 0:
                    self.train_loss_epoch.extend(self.train_loss_window)
        return loss

    def validation_step(self, batch, batch_idx):
        # ori_text, cor_text, det_labels = batch[:3]
        # outputs: 检错loss，纠错loss，检错输出，纠错输出
        with torch.no_grad():
            outputs = self.forward(*batch)
            loss = 0.  # get loss from outputs

        results = self.get_results_from_outputs(outputs, batch)
        return loss, results

    def validation_epoch_end(self, outputs):
        results = []
        
        # outputs is a list of outputs from validation_step
        for out in outputs:
            # det_acc_labels += flatten(out[1])
            # cor_acc_labels += flatten(out[2])
            results += out[-1]
        loss = np.mean([out[0] for out in outputs])
        
        # record and refresh train loss case
        if len(self.train_loss_window) > 0:
            self.train_loss_epoch.extend(self.train_loss_window)
            self.show_train_windows(n_step=0)
        if len(self.train_loss_epoch) > 0:
            train_loss_epoch = np.mean(self.train_loss_epoch) if self.train_loss_epoch else 0.
            self.logging_all(train_loss_epoch, name='train_loss_epoch')
            self.train_loss_epoch = []

        # logger info
        self.logging_all(loss, 'valid_loss')
        self.logging_all(self.current_epoch, 'epoch')

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        ret = self.validation_epoch_end(outputs, log_detail_dict=True)
        self.tb_writer.close()
        return ret

    def generate_pinyin_inputs_for_predict(self, texts):
        encoded_err = self.get_encoded_texts(texts)
        pinyin_lists = self.collator.generate_pinyin_labels(
            token_ids=encoded_err['input_ids'], texts=texts,
            similar_pinyins=False, in_batch=True)
        pinyin_inputs = torch.from_numpy(np.stack(pinyin_lists))
        return pinyin_inputs

    def predict(self, texts, detail=False, predict_shorter=False, unk_sign='֍', with_pinyin=False):
        from utils.text_utils import clean_text
        texts = [clean_text(text).lower() for text in texts]
        
        if predict_shorter:  # split long strings into substrings
            texts = [split_2_short_text(text) for text in texts]
            texts = flatten(texts)
        
        with torch.no_grad():
            inputs = self.get_encoded_texts(texts)
            pinyin_inputs = None
            if with_pinyin:
                pinyin_inputs = self.generate_pinyin_inputs_for_predict(texts)
            outputs = self.forward(texts=texts, pinyin_inputs=pinyin_inputs)

        # transform from outputs to the form of required results
        return outputs

