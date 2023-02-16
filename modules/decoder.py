import random

import numpy as np
import torch
import torch.nn as nn
from modules.multi_head_attention import MultiHeadAttention
from modules.feedforward import PoswiseFeedForwardNet
from modules.encoding import PositionalEncoding


class DecoderLayer(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v,
                 n_heads, device, attn_type, seed, few_shot):

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device,
            attn_type=attn_type, seed=seed, few_shot=few_shot)
        self.dec_enc_attn = MultiHeadAttention(
            d_model=d_model, d_k=d_k,
            d_v=d_v, n_heads=n_heads, device=device,
            attn_type=attn_type, seed=seed, few_shot=few_shot)
        self.pos_ffn = PoswiseFeedForwardNet(
            d_model=d_model, d_ff=d_ff, seed=seed)
        self.layer_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.few_shot = few_shot

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask=None, dec_enc_attn_mask=None):

        if self.few_shot:
            out, dec_self_attn, dec_loss = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        else:
            out, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        out = self.layer_norm(dec_inputs + out)
        if self.few_shot:
            out2, dec_enc_attn, dec_enc_loss = self.dec_enc_attn(out, enc_outputs, enc_outputs, dec_enc_attn_mask)
        else:
            out2, dec_enc_attn = self.dec_enc_attn(out, enc_outputs, enc_outputs, dec_enc_attn_mask)
        out2 = self.layer_norm(out + out2)
        out3 = self.pos_ffn(out2)
        out3 = self.layer_norm(out2 + out3)
        if self.few_shot:
            return out3, dec_loss, dec_enc_loss
        else:
            return out3


class Decoder(nn.Module):

    def __init__(self, d_model, d_ff, d_k, d_v,
                 n_heads, n_layers, pad_index, device,
                 attn_type, seed, few_shot):
        super(Decoder, self).__init__()

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.pad_index = pad_index
        self.device = device
        self.attn_type = attn_type
        self.pos_emb = PositionalEncoding(
            d_hid=d_model,
            device=device)
        self.layer_norm = nn.LayerNorm(d_model)
        self.layers = []
        for _ in range(n_layers):
            decoder_layer = DecoderLayer(
                d_model=d_model, d_ff=d_ff,
                d_k=d_k, d_v=d_v,
                n_heads=n_heads, device=device,
                attn_type=attn_type, seed=seed, few_shot=few_shot)
            self.layers.append(decoder_layer)
        self.layers = nn.ModuleList(self.layers)
        self.d_k = d_k
        self.few_shot = few_shot

    def forward(self, dec_inputs, enc_outputs):

        dec_outputs = self.pos_emb(dec_inputs)

        for layer in self.layers:
            if self.few_shot:
                dec_outputs, dec_loss, dec_enc_loss = layer(
                    dec_inputs=dec_outputs,
                    enc_outputs=enc_outputs,
                    dec_enc_attn_mask=None,
                )
            else:
                dec_outputs = layer(
                    dec_inputs=dec_outputs,
                    enc_outputs=enc_outputs,
                    dec_enc_attn_mask=None,
                )

        if self.few_shot:
            return dec_outputs, dec_loss, dec_enc_loss
        else:
            return dec_outputs