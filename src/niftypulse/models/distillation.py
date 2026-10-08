"""Make the model fit a latency and memory budget.

1. Distillation: a small MLP student on the *current* feature vector learns from the
   sequence teacher's softened probabilities (Hinton et al., 2015) plus the hard labels.
   No sequence window is needed at inference, which removes most of the cost.
2. Pruning: L1 unstructured magnitude pruning, then the masks are made permanent.
3. Quantization: dynamic INT8 on the exported ONNX graph (ONNX Runtime).
4. Export: ONNX for Python serving, plus a flat float32 weight file for the C++ engine.

Each stage is benchmarked (accuracy, size, latency) so the trade-off is measured, not assumed.
"""

import json
import struct
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.nn.utils import prune

from niftypulse.core.config import ModelSettings
from niftypulse.models.registry import DirectionModel, FloatArray, ModelInput, class_index, register


class StudentNet(nn.Module):
    def __init__(self, n_features: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@register
class DistilledMLP(DirectionModel):
    """Student network. `teacher_proba` must be set (aligned with train rows) before fit."""

    family = "distilled_mlp"

    def __init__(self, settings: ModelSettings) -> None:
        super().__init__(settings)
        self.net: StudentNet | None = None
        self.mu: FloatArray | None = None
        self.sd: FloatArray | None = None
        self.teacher_proba: FloatArray | None = None

    def _scale(self, x: FloatArray) -> npt.NDArray[np.float32]:
        assert self.mu is not None and self.sd is not None
        return np.clip((x - self.mu) / self.sd, -10, 10).astype(np.float32)

    def fit(
        self, train: ModelInput, y_train: FloatArray, valid: ModelInput, y_valid: FloatArray
    ) -> None:
        if self.teacher_proba is None:
            raise ValueError("set teacher_proba before fitting the student")
        s = self.settings
        torch.manual_seed(s.seed)
        x = train.matrix()
        self.mu = np.median(x, axis=0)
        q75, q25 = np.percentile(x, [75, 25], axis=0)
        self.sd = np.where(q75 - q25 > 1e-9, (q75 - q25) / 1.349, 1.0)
        xt = torch.from_numpy(self._scale(x))
        yt = torch.from_numpy(class_index(y_train))
        soft = torch.from_numpy(self.teacher_proba.astype(np.float32))
        xv = torch.from_numpy(self._scale(valid.matrix()))
        yv = torch.from_numpy(class_index(y_valid))

        self.net = StudentNet(x.shape[1], s.student_hidden)
        opt = torch.optim.AdamW(
            self.net.parameters(), lr=s.seq_learning_rate, weight_decay=s.seq_weight_decay
        )
        t, alpha = s.distill_temperature, s.distill_alpha
        best, best_state, bad = float("inf"), None, 0
        for _ in range(s.seq_epochs):
            self.net.train()
            perm = torch.randperm(len(xt))
            for i in range(0, len(perm), s.seq_batch_size):
                b = perm[i : i + s.seq_batch_size]
                logits = self.net(xt[b])
                teacher = torch.softmax(torch.log(soft[b] + 1e-7) / t, dim=1)
                kd = nn.functional.kl_div(
                    torch.log_softmax(logits / t, dim=1), teacher, reduction="batchmean"
                ) * (t * t)
                ce = nn.functional.cross_entropy(logits, yt[b])
                loss = alpha * kd + (1 - alpha) * ce
                opt.zero_grad()
                loss.backward()
                opt.step()
            self.net.eval()
            with torch.no_grad():
                val = float(nn.functional.cross_entropy(self.net(xv), yv))
            if val < best - 1e-4:
                best, best_state, bad = (
                    val,
                    {k: v.clone() for k, v in self.net.state_dict().items()},
                    0,
                )
            else:
                bad += 1
                if bad >= s.seq_patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.teacher_proba = None

    def predict_proba(self, data: ModelInput) -> FloatArray:
        assert self.net is not None
        self.net.eval()
        with torch.no_grad():
            logits = self.net(torch.from_numpy(self._scale(data.matrix())))
        return torch.softmax(logits, dim=1).numpy().astype(np.float64)

    def prune(self, amount: float) -> None:
        assert self.net is not None
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                prune.l1_unstructured(m, "weight", amount=amount)
                prune.remove(m, "weight")

    def sparsity(self) -> float:
        assert self.net is not None
        weights = [m.weight for m in self.net.modules() if isinstance(m, nn.Linear)]
        zeros = sum(int((w == 0).sum()) for w in weights)
        return zeros / sum(w.numel() for w in weights)

    def export_onnx(self, path: Path) -> None:
        assert self.net is not None and self.mu is not None
        dummy = torch.zeros(1, len(self.mu), dtype=torch.float32)
        torch.onnx.export(
            self.net,
            (dummy,),
            str(path),
            input_names=["x"],
            output_names=["logits"],
            dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )

    def export_cpp_weights(self, path: Path) -> None:
        """Flat little-endian format read by cpp/mlp_inference.hpp.

        header: magic 'NPMLP1', u32 n_layers, then per layer u32 rows, u32 cols,
        float32 weights (row-major, out x in), float32 bias; then float32 mu[F], sd[F].
        """
        assert self.net is not None and self.mu is not None and self.sd is not None
        layers = [m for m in self.net.modules() if isinstance(m, nn.Linear)]
        with path.open("wb") as f:
            f.write(b"NPMLP1")
            f.write(struct.pack("<I", len(layers)))
            for m in layers:
                w = m.weight.detach().numpy().astype("<f4")
                f.write(struct.pack("<II", *w.shape))
                f.write(w.tobytes())
                f.write(m.bias.detach().numpy().astype("<f4").tobytes())
            f.write(self.mu.astype("<f4").tobytes())
            f.write(self.sd.astype("<f4").tobytes())


def quantize_onnx(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)


@dataclass(frozen=True, slots=True)
class LatencyReport:
    name: str
    size_kb: float
    p50_us: float
    p99_us: float

    def as_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "size_kb": self.size_kb,
            "p50_us": self.p50_us,
            "p99_us": self.p99_us,
        }


def benchmark_onnx(path: Path, x: npt.NDArray[np.float32], repeats: int = 2000) -> LatencyReport:
    """Single-row latency, the shape that matters for tick-by-tick serving."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    row = x[:1]
    for _ in range(100):
        sess.run(None, {"x": row})
    times = np.empty(repeats)
    for i in range(repeats):
        t0 = time.perf_counter_ns()
        sess.run(None, {"x": row})
        times[i] = (time.perf_counter_ns() - t0) / 1e3
    return LatencyReport(
        path.name,
        path.stat().st_size / 1024,
        float(np.median(times)),
        float(np.percentile(times, 99)),
    )


def write_report(path: Path, reports: list[LatencyReport], extra: dict[str, object]) -> None:
    path.write_text(json.dumps({"latency": [r.as_dict() for r in reports], **extra}, indent=2))
