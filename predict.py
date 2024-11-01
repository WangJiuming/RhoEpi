'''
conda activate rnafm

test_path="Data/sample_seq.fasta"
model_path="Model/epoch=49-val_loss=0.11.ckpt"

python predict.py --test_path $test_path --model_path $model_path
'''

from pathlib import Path
import torch
import numpy as np
import random
from dataset import RNAModDataModule, RNAModDataset
from model import siteLevelModel
import argparse
import lightning as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from utils import read_fasta, count_pos_weight
import os
from datetime import datetime
from tqdm import tqdm
import pickle
from torch.utils.data import Dataset, DataLoader


def main():
    """read argument"""
    '''read'''
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', type=str, help='Path of the input fasta file')
    parser.add_argument('-m', '--model_path', type=str, help='Path of the pretrained model')
    parser.add_argument('-g', '--gpu', nargs='+', type=int, default=[0,1], help='GPU index, default 0, 1')
    parser.add_argument('-l', '--seq_len', default=128, type=int, help='Length of the input fasta sequence')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='Batch size, default 16')

    args = parser.parse_args()

    '''parse'''
    ''''dir related'''
    work_base = Path('.')

    test_path = Path(args.test_path)
    print(f'Getting testing data from: {test_path}')

    model_path = Path(work_base, args.model_path)
    print(f'Test model is from: {model_path}')

    result_path = Path(work_base, 'Results', f'{test_path.stem}_result.pickle')

    ''''cuda'''
    cuda_visible_devices = ','.join(map(str, args.gpu))
    print(f'cuda_visible_devices: ', cuda_visible_devices)
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_visible_devices

    ''''testing hyper param'''
    batch_size = args.batch_size
    seq_len = args.seq_len

    """init"""
    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('high')

    test_dataset = RNAModDataset(test_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    pretrained_model = siteLevelModel.load_from_checkpoint(checkpoint_path=model_path, map_location=None, seq_len=seq_len)

    """predict"""

    score_list = []
    pretrained_model.eval()

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='testing: '):
            _, x, y = batch

            y_pred = pretrained_model(x)

            y_pred = torch.sigmoid(y_pred)

            score_list.append(y_pred.view(-1).tolist())



    # Store the predictions in a list)
    with open(result_path, 'wb') as f:
        pickle.dump(score_list, f)
    print(f'Results stored at {result_path}')


if __name__ == '__main__':
    main()
