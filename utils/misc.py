import os
import torch
import numpy as np
import random
import time
import logging
import logging.handlers

THOUSAND = 1000
MILLION = 1000000


class BlackHole(object):
    def __setattr__(self, name, value):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, name):
        return self


class CheckpointManager(object):

    def __init__(self, save_dir, logger=BlackHole()):
        super().__init__()
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.ckpts = []
        self.logger = logger

        for f in os.listdir(self.save_dir):
            if f[:4] != 'ckpt':
                continue
            _, score, it = f.split('_')
            it = it.split('.')[0]
            self.ckpts.append({
                'score': float(score),
                'file': f,
                'iteration': int(it),
            })

    def get_worst_ckpt_idx(self):
        idx = -1
        worst = float('-inf')
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['score'] >= worst:
                idx = i
                worst = ckpt['score']
        return idx if idx >= 0 else None

    def get_best_ckpt_idx(self):
        idx = -1
        best = float('inf')
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['score'] <= best:
                idx = i
                best = ckpt['score']
        return idx if idx >= 0 else None
        
    def get_latest_ckpt_idx(self):
        idx = -1
        latest_it = -1
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['iteration'] > latest_it:
                idx = i
                latest_it = ckpt['iteration']
        return idx if idx >= 0 else None

    def save(self, model, args, score, others=None, step=None):

        if step is None:
            fname = 'ckpt_%.6f_.pt' % float(score)
        else:
            fname = 'ckpt_%.6f_%d.pt' % (float(score), int(step))
        path = os.path.join(self.save_dir, fname)

        torch.save({
            'args': args,
            'state_dict': model.state_dict(),
            'others': others
        }, path)

        self.ckpts.append({
            'score': score,
            'file': fname
        })

        return True

    def load_best(self):
        idx = self.get_best_ckpt_idx()
        if idx is None:
            raise IOError('No checkpoints found.')
        ckpt = torch.load(os.path.join(self.save_dir, self.ckpts[idx]['file']))
        return ckpt
    
    def load_latest(self):
        idx = self.get_latest_ckpt_idx()
        if idx is None:
            raise IOError('No checkpoints found.')
        ckpt = torch.load(os.path.join(self.save_dir, self.ckpts[idx]['file']))
        return ckpt

    def load_selected(self, file):
        ckpt = torch.load(os.path.join(self.save_dir, file))
        return ckpt


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_new_log_dir(root='./logs', postfix='', prefix=''):
    log_dir = os.path.join(root, prefix + time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()) + postfix)
    os.makedirs(log_dir)
    return log_dir


def int_tuple(argstr):
    return tuple(map(int, argstr.split(',')))


def str_tuple(argstr):
    return tuple(argstr.split(','))


def int_list(argstr):
    return list(map(int, argstr.split(',')))


def str_list(argstr):
    return list(argstr.split(','))


def log_hyperparams(writer, args):
    from torch.utils.tensorboard.summary import hparams
    vars_args = {k:v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
    exp, ssi, sei = hparams(vars_args, {})
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)
