# This code was adapted from lucidrains existing `x-transformers` repository.
from simple_parallel_transformer import Transformer, Config
from simple_parallel_transformer.autoregressive_wrapper import AutoregressiveWrapper

import random
import tqdm
import gzip
import numpy as np
import torch
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import hydra
from hydra.utils import get_original_cwd

import time
import wandb
from scipy import signal
from scipy.signal import convolve as sig_convolve

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
GENERATE_LENGTH = 512

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

@hydra.main(config_path=None, config_name="config")
def train(cfg: Config) -> None:
    set_seed(cfg.seed)

    # prepare enwik8 data

    with gzip.open(get_original_cwd() + '/./data/enwik8.gz') as file:
        X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
        trX, vaX = np.split(X, [int(90e6)])
        data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)

    class TextSamplerDataset(Dataset):
        def __init__(self, data, seq_len):
            super().__init__()
            self.data = data
            self.seq_len = seq_len

        def __getitem__(self, index):
            rand_start = torch.randint(0, self.data.size(0) - self.seq_len - 1, (1,))
            full_seq = self.data[rand_start: rand_start + self.seq_len + 1].long()
            return full_seq.cuda()

        def __len__(self):
            return self.data.size(0) // self.seq_len

    train_dataset = TextSamplerDataset(data_train, cfg.max_seq_len)
    val_dataset   = TextSamplerDataset(data_val, cfg.max_seq_len)
    train_loader  = cycle(DataLoader(train_dataset, batch_size = BATCH_SIZE))
    val_loader    = cycle(DataLoader(val_dataset, batch_size = BATCH_SIZE))

    # instantiate GPT-like decoder model

    model = Transformer(
        cfg
    )

    model = AutoregressiveWrapper(model)
    model.cuda()
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"TOTAL PARAMETERS: {pytorch_total_params}")

    wandb.init(project="transformer-enwiki8-arena", config=cfg)

    # optimizer

    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # training

    train_losses = []
    val_losses = []
    
    set_seed(cfg.seed)

    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
        start_time = time.time()
        model.train()

        train_loss = 0

        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            loss = model(next(train_loader))
            with torch.no_grad():
              train_loss += loss.item()
            loss.backward()
        train_loss = train_loss / GRADIENT_ACCUMULATE_EVERY

        end_time = time.time()
        print(f'training loss: {train_loss}')
        train_losses += [train_loss]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optim.step()
        optim.zero_grad()


        if i % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                loss = model(next(val_loader))
                print(f'validation loss: {loss.item()}')
                val_loss = loss.item()
                val_losses += [val_loss]

        if i % GENERATE_EVERY == 0:
            model.eval()
            inp = random.choice(val_dataset)[:-1]
            prime = decode_tokens(inp)
            print(f'%s \n\n %s', (prime, '*' * 100))

            sample = model.generate(inp, GENERATE_LENGTH)
            output_str = decode_tokens(sample)
            print(output_str)

        logs = {}
        
        logs = {
          **logs,
          'iter': i,
          'step_time': end_time - start_time,
          'train_loss': train_loss,
          'val_loss': val_loss,
        }
        
        wandb.log(logs)
      
    wandb.finish()

if __name__ == '__main__':
    train()
