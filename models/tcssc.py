"""
Arsitektur TCSSC:
  ToolCallEncoder → SequenceAggregator (LSTM / Transformer) → HarmClassifier
"""
import math
import torch
import torch.nn as nn
from transformers import AutoModel
from typing import Tuple

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import *


#
# KOMPONEN 1 — Tool Call Encoder
#
class ToolCallEncoder(nn.Module):
    """
    Encode satu tool call ke dense vector.
    Input : token ids dari teks "FUNC name ARGS key=val ..."
    Output: vektor (batch, embed_dim)
    """
    def __init__(self, model_name: str = ENCODER_MODEL, freeze_bert: bool = False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

        # freeze bert supaya training lebih cepat di GPU kecil
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # projection ke ukuran yang lebih kecil
        self.projection = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM // 2),
            nn.LayerNorm(EMBED_DIM // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(
        self,
        input_ids:      torch.Tensor,  # (batch, seq_len)
        attention_mask: torch.Tensor,  # (batch, seq_len)
    ) -> torch.Tensor:                 # (batch, embed_dim//2)

        out   = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # ambil [CLS] token sebagai representasi
        cls   = out.last_hidden_state[:, 0, :]
        return self.projection(cls)


#
# KOMPONEN 2A — Sequence Aggregator berbasis LSTM
#
class LSTMAggregator(nn.Module):
    """
    Membaca urutan tool call vector secara temporal dengan LSTM Bidirectional.
    Input : (batch, max_tool_calls, embed_dim//2)
    Output: (batch, lstm_hidden * 2)
    """
    def __init__(self, input_dim: int = EMBED_DIM // 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = LSTM_HIDDEN,
            num_layers  = LSTM_LAYERS,
            batch_first = True,
            bidirectional = True,
            dropout     = LSTM_DROPOUT if LSTM_LAYERS > 1 else 0,
        )
        self.layer_norm = nn.LayerNorm(LSTM_HIDDEN * 2)
        self.dropout    = nn.Dropout(LSTM_DROPOUT)

    def forward(
        self,
        x:        torch.Tensor,  # (batch, max_tool_calls, input_dim)
        seq_mask: torch.Tensor,  # (batch, max_tool_calls) bool
    ) -> torch.Tensor:           # (batch, lstm_hidden * 2)

        # hitung panjang sequence yang real (non-padding)
        lengths = seq_mask.sum(dim=1).clamp(min=1).cpu()

        # pack sequence supaya LSTM tidak proses padding
        packed   = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _   = self.lstm(packed)
        out, _   = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)

        # ambil output dari posisi terakhir yang real
        idx = (lengths - 1).clamp(min=0).to(x.device)
        idx = idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, out.size(2))
        out = out.gather(1, idx).squeeze(1)

        return self.dropout(self.layer_norm(out))


#
# KOMPONEN 2B — Sequence Aggregator berbasis Transformer
#
class PositionalEncoding(nn.Module):
    """Positional encoding standar untuk Transformer."""
    def __init__(self, d_model: int, max_len: int = MAX_TOOL_CALLS):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerAggregator(nn.Module):
    """
    Membaca urutan tool call vector dengan Transformer kecil.
    Input : (batch, max_tool_calls, embed_dim//2)
    Output: (batch, embed_dim//2)
    """
    def __init__(self, input_dim: int = EMBED_DIM // 2):
        super().__init__()
        # project input_dim ke dimensi yang divisible oleh num_heads
        self.input_proj = nn.Linear(input_dim, TRANSFORMER_FF_DIM)
        self.pos_enc    = PositionalEncoding(TRANSFORMER_FF_DIM)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = TRANSFORMER_FF_DIM,
            nhead           = TRANSFORMER_HEADS,
            dim_feedforward = TRANSFORMER_FF_DIM * 2,
            dropout         = TRANSFORMER_DROPOUT,
            batch_first     = True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=TRANSFORMER_LAYERS)
        self.layer_norm  = nn.LayerNorm(TRANSFORMER_FF_DIM)
        self.dropout     = nn.Dropout(TRANSFORMER_DROPOUT)

    def forward(
        self,
        x:        torch.Tensor,  # (batch, max_tool_calls, input_dim)
        seq_mask: torch.Tensor,  # (batch, max_tool_calls) bool — True = real
    ) -> torch.Tensor:           # (batch, transformer_ff_dim)

        x   = self.pos_enc(self.input_proj(x))
        # src_key_padding_mask: True berarti IGNORE (kebalikan dari seq_mask)
        pad_mask = ~seq_mask
        out = self.transformer(x, src_key_padding_mask=pad_mask)

        # mean pooling hanya pada posisi yang real
        mask_f = seq_mask.unsqueeze(-1).float()
        out    = (out * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)

        return self.dropout(self.layer_norm(out))


#
# KOMPONEN 3 — Harm Classifier
#
class HarmClassifier(nn.Module):
    """
    MLP dua layer untuk klasifikasi multiclass.
    Input : representasi kumulatif dari aggregator + konteks
    Output: logits (batch, num_classes)
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, CLASSIFIER_HIDDEN),
            nn.LayerNorm(CLASSIFIER_HIDDEN),
            nn.ReLU(),
            nn.Dropout(CLASSIFIER_DROPOUT),
            nn.Linear(CLASSIFIER_HIDDEN, CLASSIFIER_HIDDEN // 2),
            nn.ReLU(),
            nn.Dropout(CLASSIFIER_DROPOUT),
            nn.Linear(CLASSIFIER_HIDDEN // 2, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


#
# MODEL UTAMA — TCSSC
#
class TCSSC(nn.Module):
    """
    Tool-Call Sequence Safety Classifier.
    Menggabungkan Encoder → Aggregator → Classifier dalam satu pipeline.
    """
    def __init__(self, aggregator_type: str = "lstm", freeze_bert: bool = True):
        super().__init__()
        assert aggregator_type in ("lstm", "transformer"), "aggregator_type harus 'lstm' atau 'transformer'"
        self.aggregator_type = aggregator_type

        # komponen 1 — encoder (shared untuk tool call dan konteks)
        self.encoder     = ToolCallEncoder(freeze_bert=freeze_bert)
        self.ctx_encoder = ToolCallEncoder(freeze_bert=freeze_bert)

        enc_out_dim = EMBED_DIM // 2

        # komponen 2 — aggregator
        if aggregator_type == "lstm":
            self.aggregator = LSTMAggregator(input_dim=enc_out_dim)
            agg_out_dim     = LSTM_HIDDEN * 2
        else:
            self.aggregator = TransformerAggregator(input_dim=enc_out_dim)
            agg_out_dim     = TRANSFORMER_FF_DIM

        # fusi aggregator output + context encoding
        fused_dim = agg_out_dim + enc_out_dim

        # komponen 3 — classifier
        self.classifier = HarmClassifier(input_dim=fused_dim)

    def forward(self, batch: dict) -> torch.Tensor:
        # Encode setiap tool call
        B, T, L     = batch["input_ids"].shape       # (batch, max_tool_calls, seq_len)
        ids_flat    = batch["input_ids"].view(B * T, L)
        mask_flat   = batch["attention_mask"].view(B * T, L)

        tc_embeds   = self.encoder(ids_flat, mask_flat)   # (B*T, enc_out_dim)
        tc_embeds   = tc_embeds.view(B, T, -1)            # (B, T, enc_out_dim)

        # Aggregasi sequence
        seq_repr    = self.aggregator(tc_embeds, batch["seq_mask"])  # (B, agg_out_dim)

        # Encode konteks percakapan
        ctx_repr    = self.ctx_encoder(batch["ctx_input_ids"], batch["ctx_attention_mask"])  # (B, enc_out_dim)

        # Fusi dan klasifikasi
        fused       = torch.cat([seq_repr, ctx_repr], dim=-1)  # (B, fused_dim)
        logits      = self.classifier(fused)                    # (B, num_classes)

        return logits
