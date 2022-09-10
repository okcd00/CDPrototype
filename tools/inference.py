"""
@Time   :   2021-02-05 15:33:55
@File   :   inference.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""
import sys
import argparse
import os

import torch
from transformers import BertTokenizer

from tools.bases import args_parse
from tools.train import PretCheckpoint

sys.path.append('..')

from modeling.modeling_example import *
from utils import get_abs_path


def parse_args():
    parser = argparse.ArgumentParser(description="bbcm")
    parser.add_argument(
        "--config_file", default="csc/train_bert4csc.yml", 
        help="config file", type=str
    )
    parser.add_argument(
        "--ckpt_fn", default="epoch=2-val_loss=0.02.ckpt", 
        help="checkpoint file name", type=str
    )
    parser.add_argument("--texts", default=["马上要过年了，提前祝大家心年快乐！"], nargs=argparse.REMAINDER)
    parser.add_argument("--text_file", default='')

    args = parser.parse_args()
    return args


def load_model_directly(ckpt_file, config_file):
    # Example:
    # ckpt_fn = 'SoftMaskedBert/epoch=02-val_loss=0.02904.ckpt' (find in checkpoints)
    # config_file = 'csc/train_SoftMaskedBert.yml' (find in configs)
    
    from config import cfg

    # load cfg
    if not os.path.exists(config_file):
        config_file = get_abs_path('configs', config_file)
    cfg.merge_from_file(config_file)
    
    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_CKPT)

    # load model from checkpoint
    if os.path.exists(ckpt_file):
        cp = ckpt_file
    else:
        cp = get_abs_path('checkpoints', ckpt_file)
        
    model = CDMacBertForCsc
    model = model.load_from_checkpoint(cp,
                                       cfg=cfg,
                                       tokenizer=tokenizer)
    model.eval()
    model.to(cfg.MODEL.DEVICE)
    return model


def load_model(args):
    from config import cfg
    
    if os.path.exists(args.config_file):
        cfg.merge_from_file(args.config_file)
    else:
        cfg.merge_from_file(get_abs_path('configs', args.config_file))
    tokenizer = BertTokenizer.from_pretrained(cfg.MODEL.BERT_CKPT)

    if os.path.exists(args.ckpt_fn):
        ckpt_path = args.ckpt_fn
    else:
        file_dir = get_abs_path("checkpoints", cfg.MODEL.NAME)
        ckpt_path = os.path.join(file_dir, args.ckpt_fn)

    model = CDMacBertForCsc.load_from_checkpoint(
        ckpt_path, cfg=cfg, tokenizer=tokenizer)
    model.eval()
    model.to(cfg.MODEL.DEVICE)

    return model


def evaluate(args, test_loader='test'):
    model = load_model(args)
    model.evaluate_from_loader(loader=test_loader)


def inference(args, detail=False):
    model = load_model(args)
    texts = []
    if os.path.exists(args.text_file):
        with open(args.text_file, 'r', encoding='utf-8') as f:
            for line in f:
                texts.append(line.strip())
    else:
        texts = args.texts
    corrected_texts = model.predict(texts, detail)
    if detail:
        corrected_texts, outputs = corrected_texts
    print(corrected_texts)
    return corrected_texts


if __name__ == '__main__':
    arguments = parse_args()
    inference(arguments)
