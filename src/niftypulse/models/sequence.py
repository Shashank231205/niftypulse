"""Deep sequence models over each instrument's recent feature history (PyTorch).

Input for a row at time t is the window of the previous `seq_len` grid samples of the
same instrument, ending at t. Windows only ever contain the past, so full attention in
the transformer is not a leak. Rows without a complete, gap-free history fall back to
the training class prior.

Training: class-weighted cross-entropy (imbalance), AdamW with weight decay, dropout,
gradient clipping and early stopping on validation log-loss.
"""

import copy
import math
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from niftypulse.core.config import ModelSettings
from niftypulse.models.registry import DirectionModel, FloatArray, ModelInput, class_index, register

_NS = 1_000_000_000


def build_windows(
    data: ModelInput, seq_len: int, step_ns: int
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """(n, seq_len, F) windows aligned with data rows, plus a mask of complete windows."""
    x = data.matrix()
    n, f = x.shape
    out = np.zeros((n, seq_len, f), dtype=np.float32)
    ok = np.zeros(n, dtype=bool)
    frame = data.frame
    for _, idx in frame.groupby("instrument", sort=False).indices.items():
        idx = np.sort(idx)
        ts = frame["ts"].to_numpy()[idx]
        for j in range(seq_len - 1, len(idx)):
            span = ts[j] - ts[j - seq_len + 1]
            if span <= (seq_len - 1) * step_ns * 1.5:
                out[idx[j]] = x[idx[j - seq_len + 1 : j + 1]]
                ok[idx[j]] = True
    return out, ok


class _Standardizer:
    def __init__(self) -> None:
        self.mu: FloatArray | None = None
        self.sd: FloatArray | None = None

    def fit(self, w: npt.NDArray[np.float32]) -> None:
        flat = w.reshape(-1, w.shape[-1])
        self.mu = np.median(flat, axis=0)
        q75, q25 = np.percentile(flat, [75, 25], axis=0)
        self.sd = np.where(q75 - q25 > 1e-9, (q75 - q25) / 1.349, 1.0)

    def __call__(self, w: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        assert self.mu is not None and self.sd is not None
        return np.clip((w - self.mu) / self.sd, -10, 10).astype(np.float32)


class LSTMNet(nn.Module):
    def __init__(self, n_features: int, s: ModelSettings) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            n_features,
            s.seq_hidden,
            num_layers=s.seq_layers,
            batch_first=True,
            dropout=s.seq_dropout,
        )
        self.head = nn.Sequential(nn.Dropout(s.seq_dropout), nn.Linear(s.seq_hidden, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(out[:, -1])


class _CausalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.pad = 2 * dilation
        self.conv1 = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, 3, dilation=dilation)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.conv1(nn.functional.pad(x, (self.pad, 0))))
        y = self.drop(self.act(self.conv2(nn.functional.pad(y, (self.pad, 0)))))
        return x + y


class TCNNet(nn.Module):
    """Temporal convolutional network: causal, dilated, residual (Bai et al., 2018)."""

    def __init__(self, n_features: int, s: ModelSettings) -> None:
        super().__init__()
        self.inp = nn.Conv1d(n_features, s.seq_hidden, 1)
        levels = max(1, math.ceil(math.log2(max(s.seq_len, 2))) - 1)
        self.blocks = nn.Sequential(
            *[_CausalBlock(s.seq_hidden, 2**i, s.seq_dropout) for i in range(levels)]
        )
        self.head = nn.Linear(s.seq_hidden, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.inp(x.transpose(1, 2)))
        return self.head(h[:, :, -1])


class TransformerNet(nn.Module):
    def __init__(self, n_features: int, s: ModelSettings) -> None:
        super().__init__()
        self.proj = nn.Linear(n_features, s.seq_hidden)
        self.pos = nn.Parameter(torch.zeros(1, s.seq_len, s.seq_hidden))
        layer = nn.TransformerEncoderLayer(
            s.seq_hidden,
            s.seq_heads,
            dim_feedforward=2 * s.seq_hidden,
            dropout=s.seq_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, s.seq_layers)
        self.head = nn.Sequential(nn.LayerNorm(s.seq_hidden), nn.Linear(s.seq_hidden, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.proj(x) + self.pos[:, : x.shape[1]])
        return self.head(h[:, -1])


class _SequenceModel(DirectionModel):
    net_cls: type[nn.Module]

    def __init__(self, settings: ModelSettings) -> None:
        super().__init__(settings)
        self._net: nn.Module | None = None
        self._scaler = _Standardizer()
        self._prior = np.full(3, 1 / 3)
        self._step_ns = 0
        self.history: list[dict[str, float]] = []

    def _windows(self, data: ModelInput) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        return build_windows(data, self.settings.seq_len, self._step_ns)

    def fit(
        self, train: ModelInput, y_train: FloatArray, valid: ModelInput, y_valid: FloatArray
    ) -> None:
        s = self.settings
        torch.manual_seed(s.seed)
        torch.set_num_threads(s.n_threads)
        ts = np.sort(train.frame["ts"].unique())
        self._step_ns = int(np.median(np.diff(ts))) if len(ts) > 1 else _NS
        wt, okt = self._windows(train)
        wv, okv = self._windows(valid)
        if not okt.any() or not okv.any():
            raise ValueError(
                f"no complete {s.seq_len}-step windows in the training or early-stopping "
                "slice; the period is too short for sequence models"
            )
        yt, yv = class_index(y_train), class_index(y_valid)
        self._prior = np.bincount(yt, minlength=3) / len(yt)
        self._scaler.fit(wt[okt])
        xt, xv = torch.from_numpy(self._scaler(wt[okt])), torch.from_numpy(self._scaler(wv[okv]))
        tt, tv = torch.from_numpy(yt[okt]), torch.from_numpy(yv[okv])

        counts = np.bincount(yt[okt], minlength=3).astype(np.float32)
        weights = torch.tensor(len(tt) / (3 * np.maximum(counts, 1)), dtype=torch.float32)
        self._net = self.net_cls(xt.shape[-1], s)
        opt = torch.optim.AdamW(
            self._net.parameters(), lr=s.seq_learning_rate, weight_decay=s.seq_weight_decay
        )
        train_loss = nn.CrossEntropyLoss(weight=weights)
        eval_loss = nn.CrossEntropyLoss()
        best, best_state, bad = math.inf, None, 0
        for epoch in range(s.seq_epochs):
            self._net.train()
            perm = torch.randperm(len(xt))
            total = 0.0
            for i in range(0, len(perm), s.seq_batch_size):
                b = perm[i : i + s.seq_batch_size]
                opt.zero_grad()
                loss = train_loss(self._net(xt[b]), tt[b])
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), 1.0)
                opt.step()
                total += float(loss) * len(b)
            val = self._eval(xv, tv, eval_loss)
            self.history.append({"epoch": epoch, "train_loss": total / len(xt), "val_logloss": val})
            if val < best - 1e-4:
                best, best_state, bad = val, copy.deepcopy(self._net.state_dict()), 0
            else:
                bad += 1
                if bad >= s.seq_patience:
                    break
        if best_state is not None:
            self._net.load_state_dict(best_state)

    def _eval(self, x: torch.Tensor, y: torch.Tensor, loss: nn.Module) -> float:
        assert self._net is not None
        self._net.eval()
        with torch.no_grad():
            return float(loss(self._logits(x), y))

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        assert self._net is not None
        return torch.cat([self._net(x[i : i + 4096]) for i in range(0, len(x), 4096)])

    def predict_proba(self, data: ModelInput) -> FloatArray:
        w, ok = self._windows(data)
        out = np.tile(self._prior, (len(data), 1))
        if ok.any():
            assert self._net is not None
            self._net.eval()
            with torch.no_grad():
                logits = self._logits(torch.from_numpy(self._scaler(w[ok])))
                out[ok] = torch.softmax(logits, dim=1).numpy()
        return out

    def describe(self) -> dict[str, Any]:
        params = sum(p.numel() for p in self._net.parameters()) if self._net else 0
        return {"family": self.family, "parameters": params, "epochs": len(self.history)}


@register
class LSTMModel(_SequenceModel):
    family = "lstm"
    net_cls = LSTMNet


@register
class TCNModel(_SequenceModel):
    family = "tcn"
    net_cls = TCNNet


@register
class TransformerModel(_SequenceModel):
    family = "transformer"
    net_cls = TransformerNet
