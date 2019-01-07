# -*- coding: utf-8 -*-

import os
import shutil
import torch


def save_checkpoint(state, save_path, filename, is_best=False):
    '''
    Save model checkpoints.
    '''
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    save_file = os.path.join(save_path, filename)
    torch.save(state, save_file)

    if is_best:
        shutil.copyfile(save_file,
                        os.path.join(save_path, 'model_best.pth.tar'))
