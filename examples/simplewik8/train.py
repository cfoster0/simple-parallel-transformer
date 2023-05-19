# This code was adapted from lucidrains existing `x-transformers` repository.
from simple_parallel_transformer import Transformer, Config
from simple_parallel_transformer.autoregressive_wrapper import AutoregressiveWrapper

import random
import tqdm
import gzip
import numpy as np
import torch
from torch.optim import AdamW
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import hydra
from hydra.utils import get_original_cwd

import time
import wandb

# constants

NUM_BATCHES = int(1e5)
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 4
LEARNING_RATE = 1e-4
VALIDATE_EVERY  = 100
GENERATE_EVERY  = 500
GENERATE_LENGTH = 512
DRY_RUN = False

# helpers

def cycle(loader):
    while True:
        for data in loader:
            yield data

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens):
    return ''.join(list(map(decode_token, tokens)))

def to_dict(obj):
    d = {}
    for key in obj.keys():
        value = obj[key]
        if hasattr(value, 'keys'):
            d[key] = to_dict(value)
        else:
            d[key] = value
    return d

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

@hydra.main(config_path=None, config_name="config")
def train(cfg: Config) -> None:
    set_seed(cfg.seed)

    # prepare first 100MB of Simple English Wikipedia data

    with gzip.open(get_original_cwd() + '/./data/simplewik8.gz') as file:
        X = np.fromstring(file.read(int(95e6)), dtype=np.uint8)
        trX, vaX = np.split(X, [int(90e6)])
        data_train, data_val = torch.from_numpy(trX), torch.from_numpy(vaX)

    class TextSamplerDataset(Dataset):
        def __init__(self, data, seq_len):
            super().__init__()
            self.data = data
            self.seq_len = seq_len

        def __getitem__(self, index):
            rand_start = torch.randint(0, self.data.size(0) - self.seq_len, (1,))
            full_seq = self.data[rand_start: rand_start + self.seq_len].long()
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
    
    config = to_dict(cfg)
    config['parameters'] = pytorch_total_params

    if not DRY_RUN:
        wandb.init(project="transformer-simplewik8-interp", config=config)

    # optimizer

    optim = AdamW([
      {
        'params': [param for name, param in model.named_parameters() if (('in_proj' in name) or ('out_proj' in name))],
        'weight_decay': 0.01,
      },
      {
        'params': [param for name, param in model.named_parameters() if (('in_proj' not in name) and ('out_proj' not in name))],
        'weight_decay': 0.,
      },
    ], lr=LEARNING_RATE, weight_decay=0.01)

    # training

    train_losses = []
    val_losses = []
    tokens = 0
    
    set_seed(cfg.seed)

    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
        start_time = time.time()
        model.train()

        train_loss = 0

        logs = {}

        for name, param in model.named_parameters():
          if param.numel() == 0:
            continue
          logs[f"Parameters - {name} - MEAN"] = param.mean().item()
          logs[f"Parameters - {name} - STDEV"] = param.std().item()
          logs[f"Parameters - {name} - MIN"] = param.min().item()
          logs[f"Parameters - {name} - MAX"] = param.max().item()

        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            loss = model(next(train_loader))
            with torch.no_grad():
              train_loss += loss.item()
            loss.backward()
            tokens += cfg.max_seq_len * GRADIENT_ACCUMULATE_EVERY
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
            inp = random.choice(val_dataset)
            prime = decode_tokens(inp)
            print(f'%s \n\n %s', (prime, '*' * 100))

            sample = model.generate(inp, GENERATE_LENGTH)
            output_str = decode_tokens(sample)
            print(output_str)

        
        logs = {
          **logs,
          'tokens': tokens,
          'iter': i,
          'step_time': end_time - start_time,
          'train_loss': train_loss,
          'val_loss': val_loss,
        }
        
        if not DRY_RUN:
            wandb.log(logs)
      
    if not DRY_RUN:
        wandb.finish()

if __name__ == '__main__':
    train()
