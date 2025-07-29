"""Event Transformer (v6 – label required, JSON-friendly loader)
============================================================
Dataset layout (per-sample JSON object)
--------------------------------------
```jsonc
{
  "events": [[f1,f2,f3], ...],   // list length M, each vec len 3
  "y":       [1024 floats],      // **one vector per sequence**
  "h":       [[h-vec 1024]*M],   // list length M, each vec len 1024
  "label":   true | false        // REQUIRED boolean
}
```
"""

from __future__ import annotations
from typing import List, Dict, Tuple
import json, math, pathlib

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch.nn.functional as F

MAX_SEQ_LEN = 10  # max length of event sequence, used in PositionalEncoding

###############################################################################
# Data utilities
###############################################################################
import math
import torch
from torch.utils.data import DataLoader, random_split

def make_loaders(
    json_path: str,
    batch_size: int = 128,
    train_frac: float = 0.8,
    *,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    Reads the JSON file, shuffles deterministically, and returns
    (train_loader, eval_loader).

    Args
    ----
    json_path   : path to the dataset JSON
    batch_size  : mini-batch size for both splits
    train_frac  : proportion of samples that go into the training set
    seed        : random seed so you can reproduce the same split
    """
    # 1 ─ read the full dataset
    full_ds = EventSequenceDataset.from_json(json_path)

    # 2 ─ compute split lengths
    n_total       = len(full_ds)
    n_train       = math.floor(n_total * train_frac)
    n_eval        = n_total - n_train
    lengths       = (n_train, n_eval)

    # 3 ─ reproducible random split
    gen = torch.Generator().manual_seed(seed)
    train_ds, eval_ds = random_split(full_ds, lengths, generator=gen)

    # 4 ─ wrap each subset in a DataLoader
    kwargs = dict(batch_size=batch_size,
                  collate_fn=EventSequenceDataset.collate,
                  num_workers=0,   # or >0 if you want background workers
                  pin_memory=True)

    return (
        DataLoader(train_ds, shuffle=True,  **kwargs),
        DataLoader(eval_ds,  shuffle=True, **kwargs),
    )

class EventSequenceDataset(Dataset):
    """Converts raw JSON-like dicts to tensors.

    Required keys per sample:
        • "events" / "e" : list[list[float]]  shape (T, event_dim)
        • "y"            : list[float]        shape (y_dim,)
        • "h"            : list[list[float]]  shape (T, h_dim)
        • "label"        : bool
    """

    def __init__(self, raw_samples: List[Dict]):
        self.samples: List[Dict[str, Tensor]] = []
        for ix, s in enumerate(raw_samples):
            e_key = "events" if "events" in s else "e"
            if e_key not in s:
                raise KeyError(f"Sample {ix} missing 'events' key")
            if "y" not in s or "h" not in s or "success" not in s:
                raise KeyError(f"Sample {ix} missing one of required keys 'y', 'h', 'label'")

            e = torch.tensor(s[e_key], dtype=torch.float32)        # (T, event_dim)
            h = torch.tensor(s["h"], dtype=torch.float32)       # (T, h_dim)
            y_seq = torch.tensor(s["y"], dtype=torch.float32)      # (y_dim,)
            if y_seq.ndim != 1:
                raise ValueError("'y' must be 1-D vector")
            label = torch.tensor(float(s["success"]), dtype=torch.float32)  # 0.0 / 1.0 , note: label called success in the original JSON

            self.samples.append({"e": e, "y": y_seq, "h": h, "label": label})

    # ------------------------------------------------------------------
    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> "EventSequenceDataset":
        """Load a single JSON file containing an array of samples."""
        with open(path, "r", encoding="utf‑8") as f:
            raw = json.load(f)  # expecting list[dict]
            if not isinstance(raw, list):
                raise ValueError("JSON must be an array of sample objects")
        return cls(raw)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]

    # ------------------------------------------------------------------
    @staticmethod
    def collate(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        
        lengths = torch.tensor([s["e"].size(0) for s in batch])
        max_len = MAX_SEQ_LEN
        event_dim = batch[0]["e"].size(1)
        h_dim     = batch[0]["h"].size(1)

        e_seqs, h_seqs, y_seqs, labels = [], [], [], []
        for s in batch:
            T = s["e"].size(0)
            pad_e = torch.zeros((max_len - T, event_dim))
            pad_h = torch.zeros((max_len - T, h_dim))

            e_seqs.append(torch.cat([s["e"], pad_e]))
            h_seqs.append(torch.cat([s["h"], pad_h]))
            y_repeat = s["y"].unsqueeze(0).expand(max_len, -1)  # repeat y for each time step
            y_seqs.append(y_repeat) 
            labels.append(s["label"])  # scalar tensor

        return {
            "e": torch.stack(e_seqs),               # (B, T_max, event_dim)
            "h": torch.stack(h_seqs),               # (B, T_max+1, h_dim)
            "y": torch.stack(y_seqs),                    # (B, y_dim)
            "lengths": lengths,                     # (B,)
            "label": torch.stack(labels),           # (B,)
        }

###############################################################################
# Positional Encoding (unchanged)
###############################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[:, : x.size(1)]

###############################################################################
# Event Transformer (unchanged core)
###############################################################################
class EventTransformer(pl.LightningModule):
    def __init__(
        self,
        *,
        event_dim: int,
        y_dim: int,
        h_dim: int,
        model_dim: int = 128,
        num_layers: int = 4,
        nhead: int = 8,
        dropout: float = 0.1,
        lr: float = 1e-4,
        cls_loss_weight: float = 1.0,
        recon_loss: str = "cosine",
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        D = model_dim

        self.e_proj = nn.Linear(event_dim, D)
        self.h_proj = nn.Linear(h_dim,   D)
        self.y_proj = nn.Linear(y_dim,   D)
        self.fuse   = nn.Linear(D * 3,   D)
        self.pos_enc = PositionalEncoding(D, max_len=MAX_SEQ_LEN)

        enc_layer = nn.TransformerEncoderLayer(d_model=D, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.next_state_head = nn.Linear(D, h_dim)
        
        #self.register_buffer("causal_base", self._causal_mask(MAX_SEQ_LEN, device=self.device))

        hidden_dim = max(D // 2, 4)
        self.cls_head = nn.Sequential(
            nn.Linear(D, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        self.recon_loss = recon_loss.lower()
        if self.recon_loss not in {"mse", "cosine"}:
            raise ValueError("recon_loss must be 'mse' or 'cosine'")
        self.bce  = nn.BCEWithLogitsLoss()
        self.cls_w = cls_loss_weight
        self.lr   = lr

    # ------------------------------------------------------------------
    def _causal_mask(
        self,
        T: int,
        ref: torch.Tensor,
        mode: str = "diag",   # "causal" or "diag"
    ) -> torch.Tensor:
        """
        Returns an additive attention mask (−∞ for disallowed positions, 0 otherwise)
        that is safe for fp16/bf16.  
        mode = "causal":   standard upper-triangular mask  
        mode = "diag" :    diagonal-only mask (each query attends only to itself)
        """
        bad = torch.finfo(ref.dtype).min        # −65504 fp16, −3.4e38 fp32
        m   = torch.full((T, T), bad, dtype=ref.dtype, device=ref.device)

        if mode == "causal":
            # keep 0 on / below diagonal
            m.triu_(diagonal=1)
        elif mode == "diag":
            # allow only self-attention
            m.fill_diagonal_(0.)
        else:
            raise ValueError(f"unknown mask mode: {mode}")

        return m

    def _prepare_src(self, e: Tensor, y: Tensor, h: Tensor) -> Tensor:
        
        emb_e = self.e_proj(e) # (B,T,D)
        # h: (B,1,h_dim) → (B,T,D)
        if h.ndim == 2:  # single h vector per batch
            h = h.unsqueeze(1).expand(-1, e.size(1), -1)
        emb_h = self.h_proj(h) # (B,T,D)
        emb_y = self.y_proj(y) # (B,T,D)
        fused = self.fuse(torch.cat([emb_e, emb_y, emb_h], dim=-1))
        return self.pos_enc(fused)

    def _rollout(self, e: Tensor, y: Tensor) -> Tensor:
        """
        Autoregressively predict hₜ from t = 0 … T-1, starting from zeros.
        Returns tensor (B, T, h_dim)
        """
        B, T, _ = e.shape
        h_dim = self.h_proj.in_features          # same as self.hparameters["h_dim"]
        prev_h = e.new_zeros(B, h_dim)           # initial h₀ = 0
        preds = []

        for t in range(T):
            
            src  = self._prepare_src(e[:, :t+1], y[:, :t+1], prev_h)  # (B, t+1, D)
            if torch.isnan(src).any():
                raise RuntimeError(f"NaN in src at step {t}") 
            causal = self._causal_mask(t+1, src)
            mem  = self.encoder(src, mask=causal)
            # Aggregate over time dimension
            mem = mem.mean(dim=1)                  # (B,D)
            next_h = self.next_state_head(mem)  # (B,h_dim)
            preds.append(next_h)
            prev_h = next_h.detach()             # or keep grads for full TBPTT
            
        return torch.stack(preds, dim=1)         # (B,T,h_dim)

    # ------------------------------------------------------------------
    def _step(self, batch):
        
        e, y, target_h = batch["e"], batch["y"], batch["h"]   # h already length T
        lengths = batch["lengths"]
        pred_h = self._rollout(e, y)                     # (B,T,h_dim)
        tgt_h    = target_h[:, :MAX_SEQ_LEN]             # (B,T,h_dim)
        
        if self.recon_loss == "mse":
            loss_recon = F.mse_loss(pred_h, tgt_h)       # identical to before
        else:                                            # cosine version
            # Cosine similarity per (B,T); convert to a *loss* in [0,2]
            cos = F.cosine_similarity(pred_h, tgt_h, dim=-1)  # (B,T)
            loss_recon = (1.0 - cos).mean()
        # ────────────────────────────────────────────────────────────────
        label   = batch["label"]
        # pick the *last* timestep’s hidden for classification
        last_hidden = pred_h.gather(
            1,
            (lengths-1).unsqueeze(-1).unsqueeze(-1).expand(-1,1,target_h.size(-1))
        ).squeeze(1)
        
        pred_cls = self.cls_head(self.h_proj(last_hidden))    # reuse existing head
        loss_cls  = self.bce(pred_cls, label.unsqueeze(1))
        loss = loss_recon + self.cls_w * loss_cls
        return loss, loss_recon, loss_cls


    # ------------------------------------------------------------------
    def training_step(self, batch: Dict[str, Tensor], batch_idx: int):
        loss, mse, bce = self._step(batch)
        self.log_dict({"train_loss": loss, "train_mse": mse, "train_bce": bce}, prog_bar=True)
        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int):
        loss, mse, bce = self._step(batch)
        self.log_dict({"val_loss": loss, "val_mse": mse, "val_bce": bce}, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

###############################################################################
# Train
###############################################################################
if __name__ == "__main__":
    train_loader, val_loader = make_loaders(
        "sequential_tasks/data/dataset.json",
        batch_size=128,
        train_frac=0.8,   # 80 % train / 20 % val
    )

    model = EventTransformer(event_dim=3, y_dim=1024, h_dim=1024)
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="auto",
        log_every_n_steps=10,
    )
    trainer.fit(model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)
