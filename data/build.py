"""
@Time   :   2021-01-21 14:20:50
@File   :   build.py
@Author :   Abtion
@Email  :   abtion{at}outlook.com
"""

from utils import get_abs_path
from data.loaders import get_csc_loader


def get_train_loader(cfg, get_loader_fn=None, ep=0, **kwargs):
    # single function for changing from different datasets.
    if ep > 0:
        path = get_abs_path(cfg.DATASETS.TRAIN) + f".ep{ep}"
        print(f"Now loading dataset from path: {path}")
    else:
        path = get_abs_path(cfg.DATASETS.TRAIN)
    if get_loader_fn is None:
        get_loader_fn = get_csc_loader
    train_loader = get_loader_fn(path,
                                 batch_size=cfg.SOLVER.BATCH_SIZE,
                                 shuffle=True,
                                 num_workers=cfg.DATALOADER.NUM_WORKERS,
                                 **kwargs)
    return train_loader


def make_loaders(cfg, get_loader_fn=None, **kwargs):

    if get_loader_fn is None:
        get_loader_fn = get_csc_loader

    if cfg.DATASETS.TRAIN == '':
        train_loader = None
    else:
        train_loader = get_loader_fn(get_abs_path(cfg.DATASETS.TRAIN),
                                     batch_size=cfg.SOLVER.BATCH_SIZE,
                                     shuffle=False,  # shuffle takes a long list of indexes
                                     num_workers=cfg.DATALOADER.NUM_WORKERS,
                                     **kwargs)
    if cfg.DATASETS.VALID == '':
        valid_loader = None
    else:
        valid_loader = get_loader_fn(get_abs_path(cfg.DATASETS.VALID),
                                     batch_size=cfg.TEST.BATCH_SIZE,
                                     shuffle=False,
                                     num_workers=cfg.DATALOADER.NUM_WORKERS,
                                     **kwargs)
    if cfg.DATASETS.TEST == '':
        test_loader = None
    else:
        test_loader = get_loader_fn(get_abs_path(cfg.DATASETS.TEST),
                                    batch_size=cfg.TEST.BATCH_SIZE,
                                    shuffle=False, 
                                    num_workers=cfg.DATALOADER.NUM_WORKERS,
                                    **kwargs)
    return train_loader, valid_loader, test_loader
