from tqdm import tqdm
import glob
import os
import numpy as np
from prefetch_generator import BackgroundGenerator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from ops import positional_encoding

from util import to_device, plot_att_heads
from model import Encoder, Decoder
from dataset import Dataset, _symbol_to_id, parse_text
from audio_process import MelWav, sample_rate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epoch_total = 64
batch_size = 32
enc_lr = 0.0001
dec_lr = 0.0005
emb_lr = 0.0001

# -----------------------------------

text_embedding = nn.Embedding(num_embeddings=len(_symbol_to_id), embedding_dim=256).to(device)
pos_embedding = nn.Embedding.from_pretrained(positional_encoding(512, 256), freeze=True).to(device)
pos_embedding_ = nn.Embedding.from_pretrained(positional_encoding(256, 256), freeze=True).to(device)

encoder = Encoder(emb_channels=256).to(device)
decoder = Decoder(mel_channels=80, enc_channels=256, emb_channels=256).to(device)
mel_to_wav = MelWav().to(device)

optimizer = torch.optim.Adam([{'params': text_embedding.parameters(), 'lr': emb_lr},
                              {'params': encoder.parameters(), 'lr': enc_lr},
                              {'params': decoder.parameters(), 'lr': dec_lr}],
                             lr=0.001)

# -----------------------------------

logs_idx = f'emb_lr{emb_lr}-enc_lr{enc_lr}-dec_lr{dec_lr}-batch_size{batch_size}'
saves = glob.glob(f'logs/{logs_idx}/*.pt')

saves.sort(key=os.path.getmtime)
checkpoint = torch.load(saves[-1], )
text_embedding.load_state_dict(checkpoint['text_embedding'])
text_embedding.eval()
encoder.load_state_dict(checkpoint['encoder'])
encoder.eval()
decoder.load_state_dict(checkpoint['decoder'])
decoder.eval()

with torch.no_grad():
    text_data = parse_text('hello, this is just a test').to(device)
    text_data = text_data.unsqueeze(0)
    text_emb = text_embedding(text_data)

    text_pos = (torch.arange(text_data.size(1)) + 1).to(device)
    text_pos = text_pos.unsqueeze(0).to(device)
    text_pos_emb = pos_embedding_(text_pos)
    text_mask = (text_pos == 0).unsqueeze(1)
    enc_out, att_heads_enc = encoder(text_emb, text_mask, text_pos_emb)

    mel_pos = torch.arange(1, 512).view(1, 511).to(device)
    mel_pos_emb_ = pos_embedding(mel_pos)
    mel_mask_ = torch.triu(torch.ones(511, 511, dtype=torch.bool), 1).unsqueeze(0).to(device)
    # [B, T, C], [B, T, C], [B, T, 1], [B, T, T_text]
    mel = torch.zeros(1, 511, 80).to(device)
    for pos_idx in tqdm(range(511)):
        mel_pos_emb = mel_pos_emb_[:, :pos_idx + 1]
        mel_mask = mel_mask_[:, :pos_idx + 1, :pos_idx + 1]
        mels_out, mels_out_post, gates_out, att_heads_dec, att_heads = decoder(mel[:, :pos_idx + 1], enc_out,
                                                                               mel_mask, text_mask, mel_pos_emb)

        mel[:, pos_idx] = mels_out_post[:, pos_idx]
        if gates_out[0, -1, 0] > .5:
            mel = mel[:, :pos_idx + 1]
            break
wav = mel_to_wav(mel[0])
torchaudio.save('test.wav', wav.to('cpu'), sample_rate, 32)
