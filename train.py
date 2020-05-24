from tqdm import tqdm
import glob
import os
import numpy as np
from prefetch_generator import BackgroundGenerator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils import tensorboard
from torchvision import utils
from ops import positional_encoding

from util import to_device, plot_att_heads
from model import Encoder, Decoder
from dataset import Dataset, _symbol_to_id
from audio_process import MelWav, sample_rate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epoch_total = 256
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

optimizer = torch.optim.Adam([{'params': text_embedding.parameters(), 'lr': emb_lr},
                              {'params': encoder.parameters(), 'lr': enc_lr},
                              {'params': decoder.parameters(), 'lr': dec_lr}],
                             lr=0.001)

# -----------------------------------

logs_idx = f'emb_lr{emb_lr}-enc_lr{enc_lr}-dec_lr{dec_lr}-batch_size{batch_size}'
saves = glob.glob(f'logs/{logs_idx}/*.pt')
dataset = Dataset('../DATASETS/LJSpeech-1.1/metadata.csv', '../DATASETS/LJSpeech-1.1')
dataloader = DataLoader(dataset, collate_fn=dataset.collocate, batch_size=batch_size,
                        shuffle=True, num_workers=0, drop_last=True)
writer = tensorboard.SummaryWriter(log_dir=f'logs/{logs_idx}')
mel_to_wav = MelWav().to(device)
if len(saves) != 0:
    saves.sort(key=os.path.getmtime)
    checkpoint = torch.load(saves[-1], )
    text_embedding.load_state_dict(checkpoint['text_embedding'])
    text_embedding.train()
    encoder.load_state_dict(checkpoint['encoder'])
    encoder.train()
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.train()
    optimizer.load_state_dict(checkpoint['optimizer'])

    epoch = checkpoint['epoch']
    global_idx = checkpoint['global_idx']
else:
    epoch = 0
    global_idx = 0


# ---------------------------------------

summ_counter = 0
mean_losses = np.zeros(4)
mean_metrics = np.zeros(4)
for epoch in tqdm(range(epoch, epoch_total),
                  initial=epoch, total=epoch_total, leave=False, dynamic_ncols=True):
    for idx, batch in enumerate(tqdm(BackgroundGenerator(dataloader),
                                     total=len(dataloader), leave=False, dynamic_ncols=True)):
        text_data, text_pos, text_mask, mel_data, mel_pos, mel_mask, gate = to_device(batch, device)

        # audio_data = F.avg_pool1d(audio_data, kernel_size=2, padding=1)
        text_emb = text_embedding(text_data)
        text_pos_emb = pos_embedding_(text_pos)
        enc_out, att_heads_enc = encoder(text_emb, text_mask, text_pos_emb)

        mel_pos_emb = pos_embedding(mel_pos)
        # [B, T, C], [B, T, C], [B, T, 1], [B, T, T_text]
        mels_out, mels_out_post, gates_out, att_heads_dec, att_heads = decoder(mel_data, enc_out,
                                                                               mel_mask, text_mask, mel_pos_emb)
        text_len = text_pos.max(1)[0]
        mel_len = mel_pos.max(1)[0]
        loss_mel = torch.sum((mels_out - mel_data) ** 2) / torch.sum(mel_len * text_len)
        loss_mel_post = torch.sum((mels_out_post - mel_data) ** 2) / torch.sum(mel_len * text_len)
        loss_gate = F.binary_cross_entropy(gates_out, gate)
        loss = loss_mel + loss_mel_post + loss_gate

        optimizer.zero_grad()
        loss.backward()

        grad_norm_enc = nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        grad_norm_dec = nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)

        optimizer.step()

        # -----------------------------------------

        global_idx += 1
        summ_counter += 1
        mean_losses += [loss_mel.item(),
                        loss_mel_post.item(),
                        loss_gate.item(),
                        loss.item()]
        mean_metrics += [grad_norm_enc,
                         grad_norm_dec,
                         encoder.pos_alpha.item(),
                         decoder.pos_alpha.item()]

        if global_idx % 100 == 0:
            # print(attentions_t[:4])
            mean_losses /= summ_counter
            mean_metrics /= summ_counter
            writer.add_scalar('loss/mel', mean_losses[0], global_idx)
            writer.add_scalar('loss/mel_post', mean_losses[1], global_idx)
            writer.add_scalar('loss/gate', mean_losses[2], global_idx)
            writer.add_scalar('loss_total', mean_losses[3], global_idx)

            writer.add_scalar('grad_norm/enc', mean_metrics[0], global_idx)
            writer.add_scalar('grad_norm/dec', mean_metrics[1], global_idx)
            writer.add_scalar('alpha/enc', mean_metrics[2], global_idx)
            writer.add_scalar('alpha/dec', mean_metrics[3], global_idx)
            mean_losses = np.zeros(4)
            mean_metrics = np.zeros(4)
            summ_counter = 0

            if global_idx % 1000 == 0:
                writer.add_audio(f'audio/target', mel_to_wav(mel_data[0, :mel_len[0]]),
                                 global_step=global_idx, sample_rate=sample_rate)
                writer.add_audio(f'audio/out', mel_to_wav(mels_out[0, :mel_len[0]]),
                                 global_step=global_idx, sample_rate=sample_rate)

            mel_data = mel_data.unsqueeze(1).transpose(2, 3)
            mel_data = utils.make_grid(mel_data[:4], nrow=1, padding=2, pad_value=1, normalize=True, scale_each=True)
            # writer.add_image(f'mel/target', mel_data, global_idx)

            mels_out = mels_out.unsqueeze(1).transpose(2, 3)
            mels_out = utils.make_grid(mels_out[:4], nrow=1, padding=2, pad_value=1,
                                       normalize=True, scale_each=True)

            mels_out_post = mels_out_post.unsqueeze(1).transpose(2, 3)
            mels_out_post = utils.make_grid(mels_out_post[:4], nrow=1, padding=2, pad_value=1,
                                            normalize=True, scale_each=True)
            # writer.add_image(f'mel/post_prediction', mels_out_post, global_idx)
            writer.add_image(f'mel/target---mel_out---mel_out_post',
                             torch.cat([mel_data, mels_out, mels_out_post], 2), global_idx)

            gate = gate.expand(-1, -1, 10).unsqueeze(1).transpose(2, 3)
            gates_out = gates_out.expand(-1, -1, 10).unsqueeze(1).transpose(2, 3)
            gate = torch.cat([gate, gates_out], 2)
            gate = utils.make_grid(gate[:4], nrow=1, padding=2, pad_value=.5,
                                   normalize=True, range=(0, 1))
            # writer.add_image(f'mel/post_prediction', mels_out_post, global_idx)
            writer.add_image(f'gate', gate, global_idx)

            writer.add_image(f'attention', plot_att_heads(att_heads, 1), global_idx)
            writer.add_image(f'attention/enc', plot_att_heads(att_heads_enc, 1), global_idx)
            writer.add_image(f'attention/dec', plot_att_heads(att_heads_dec, 1), global_idx)
            # ---------------------------------------------
            if global_idx % 1000 == 0:
                with torch.no_grad():
                    mel_pos = torch.arange(1, 512).view(1, 511).expand(4, -1).to(device)
                    mel_pos_emb = pos_embedding(mel_pos)
                    mel_mask = torch.triu(torch.ones(511, 511, dtype=torch.bool), 1).unsqueeze(0).to(device)
                    mel_data_ = torch.zeros(4, 511, 80).to(device)
                    enc_out = enc_out[:4]
                    text_mask = text_mask[:4]
                    for pos_idx in tqdm(range(511), leave=False, dynamic_ncols=True):
                        (mels_out, mels_out_post_,
                         gates_out, att_heads_dec, att_heads) = decoder(mel_data_[:, :pos_idx + 1], enc_out,
                                                                        mel_mask[:, :pos_idx + 1, :pos_idx + 1],
                                                                        text_mask, mel_pos_emb[:, :pos_idx + 1])

                        mel_data_[:, pos_idx] = mels_out_post_[:, pos_idx]
                        if torch.sum(torch.sum(gates_out > .5, dim=1) > 0) == 4:
                            mel_data_ = mel_data_[:, :pos_idx + 1]
                            break
                # [B, T, 1]
                writer.add_audio(f'audio/pred', mel_to_wav(mels_out[0]),
                                 global_step=global_idx, sample_rate=sample_rate)
                gate = torch.ones(4, mel_data_.size(1), 1)
                for gate_idx, gate_out in enumerate(gates_out):
                    gate_start = (gate_out > .5).nonzero()
                    if gate_start.size(0) != 0:
                        gate[gate_idx] = torch.arange(gate_out.size(0)).unsqueeze(1) <= gate_start[0, 0]
                gate = gate.to(device)
                mel_data_ = 0.5 * (gate + 1) * mel_data_
                mel_data_ = mel_data_.unsqueeze(1).transpose(2, 3)
                mel_data_ = utils.make_grid(mel_data_, nrow=1, padding=2, pad_value=1, normalize=True, scale_each=True)
                writer.add_image(f'test/mel', torch.cat([mel_data, mel_data_], 2), global_idx)
                writer.add_image(f'test/attention', plot_att_heads(att_heads, 1), global_idx)
                writer.add_image(f'test/attention/enc', plot_att_heads(att_heads_enc, 1), global_idx)
                writer.add_image(f'test/attention/dec', plot_att_heads(att_heads_dec, 1), global_idx)
                del mel_data_, mels_out_post_

    # ---------------------------------------------

    saves = glob.glob(f'logs/{logs_idx}/*.pt')
    if len(saves) == 10:
        saves.sort(key=os.path.getmtime)
        os.remove(saves[0])

    torch.save({
        'epoch': epoch + 1,
        'global_idx': global_idx,
        'text_embedding': text_embedding.state_dict(),
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer': optimizer.state_dict()},
        f'logs/{logs_idx}/model_{epoch + 1}.pt')

    # check for early exit
    with open('run.txt', 'r+') as run:
        if not int(run.read()):
            run.seek(0)
            run.write('1')
            exit()
