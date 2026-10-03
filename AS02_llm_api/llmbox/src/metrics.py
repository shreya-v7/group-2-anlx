"""
training_metrics.py

Dependency-light training metrics for LLM jobs. No MLflow, no UI, no network.

Hard dependencies: none (standard library only).
Optional, each one just fills in more metrics:
    psutil          -> memory + CPU metrics
    nvidia-ml-py    -> GPU utilization / memory / power (imported as `pynvml`)
    torch           -> parameter count fallback, GPU count fallback
    matplotlib      -> loss-curve PNG in the saved report

Usage (Hugging Face Trainer):

    tracker = MetricsTracker(cfg)
    tracker.start()
    trainer.train()
    metrics = tracker.stop(trainer)          # also sets cfg.metrics
    print(format_report(metrics))
    save_report(metrics, cfg.output_dir, trainer)

Usage (custom loop): call tracker.add_tokens(n) each batch and pass
tracker.stop(model=model, loss=last_loss, eval_loss=last_eval_loss).
"""
import json
import math
import os
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import psutil
except ImportError:  # optional
    psutil = None

# --------------------------------------------------------------------------
# Defaults (all overridable)
# --------------------------------------------------------------------------
DEFAULT_TOKEN_BUDGET = 1_000_000     # per-job budget shown to students
DEFAULT_GPU_WATTS = 300.0            # assumed per-GPU draw if power can't be measured
DEFAULT_CPU_WATTS = 65.0             # assumed CPU package draw
DEFAULT_CARBON_INTENSITY = 0.4       # kg CO2e per kWh (rough global grid average)
A100_EFFECTIVE_FLOPS = 1.25e14       # ~40% of an A100's 312 TFLOP/s bf16 peak
GPT3_TRAIN_FLOPS = 3.14e23
KG_CO2_PER_KM_DRIVEN = 0.25          # average gasoline passenger car


# --------------------------------------------------------------------------
# Data container (scalars only, so it is safe to store on an OmegaConf Config)
# --------------------------------------------------------------------------
@dataclass
class TrainingMetrics:
    """Metrics and resource use for one training job."""
    # Timing
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    elapsed_time: Optional[float] = None            # seconds
    # Memory (host process RSS, sampled during the job)
    max_memory_bytes: Optional[int] = None
    min_memory_bytes: Optional[int] = None
    median_memory_bytes: Optional[int] = None
    total_memory_bytes: Optional[int] = None        # total system RAM
    memory_used_percent: Optional[float] = None     # peak / total * 100
    # Utilization (averages over the job)
    avg_cpu_util_percent: Optional[float] = None
    avg_gpu_util_percent: Optional[float] = None
    # Compute
    flops_estimate: Optional[float] = None
    flops_level: Optional[int] = None               # 1 (very light) .. 5 (extreme)
    flops_explanation: Optional[str] = None
    # Energy / carbon
    energy_kwh: Optional[float] = None
    carbon_kg_estimate: Optional[float] = None
    carbon_explanation: Optional[str] = None
    # Tokens
    tokens_processed: int = 0
    token_budget: int = DEFAULT_TOKEN_BUDGET
    tokens_source: Optional[str] = None             # how tokens_processed was obtained
    # Performance
    loss: Optional[float] = None
    perplexity: Optional[float] = None
    eval_loss: Optional[float] = None
    eval_perplexity: Optional[float] = None

    @property
    def tokens_remaining(self) -> int:
        return max((self.token_budget or 0) - (self.tokens_processed or 0), 0)

    @property
    def token_budget_used_percent(self) -> Optional[float]:
        if not self.token_budget:
            return None
        return 100.0 * (self.tokens_processed or 0) / self.token_budget

    @property
    def over_budget(self) -> bool:
        return bool(self.token_budget) and (self.tokens_processed or 0) > self.token_budget

    def to_grouped_dict(self) -> Dict[str, Dict[str, Any]]:
        econ = ("elapsed_time", "max_memory_bytes", "min_memory_bytes",
                "median_memory_bytes", "total_memory_bytes", "memory_used_percent",
                "avg_cpu_util_percent", "avg_gpu_util_percent", "flops_estimate",
                "flops_level", "flops_explanation", "energy_kwh", "carbon_kg_estimate",
                "carbon_explanation", "token_budget_used_percent", "tokens_processed",
                "tokens_remaining", "token_budget", "tokens_source")
        perf = ("loss", "perplexity", "eval_loss", "eval_perplexity")
        return {
            "LLM Training Job Economic Metrics": {k: getattr(self, k) for k in econ},
            "LLM Training Performance": {k: getattr(self, k) for k in perf},
        }


# --------------------------------------------------------------------------
# Background sampler for memory / CPU / GPU
# --------------------------------------------------------------------------
def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


class ResourceMonitor:
    """Samples resource use on a background thread every `interval` seconds."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.rss: List[int] = []
        self.cpu: List[float] = []
        self.gpu_util: List[float] = []
        self.gpu_mem: List[int] = []
        self.gpu_power: List[float] = []          # watts, summed over devices
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc = None
        self._nvml = None
        self._handles: list = []

    def _init_nvml(self) -> None:
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i)
                             for i in range(pynvml.nvmlDeviceGetCount())]
        except Exception:
            self._nvml, self._handles = None, []

    def _sample(self, cpu: bool = True) -> None:
        if psutil is not None:
            self.rss.append(self._proc.memory_info().rss)
            if cpu:
                self.cpu.append(psutil.cpu_percent(interval=None))
        if self._handles:
            nv = self._nvml
            utils = [u for u in (_safe(lambda h=h: nv.nvmlDeviceGetUtilizationRates(h).gpu)
                                 for h in self._handles) if u is not None]
            mems = [x for x in (_safe(lambda h=h: nv.nvmlDeviceGetMemoryInfo(h).used)
                                for h in self._handles) if x is not None]
            pows = [x for x in (_safe(lambda h=h: nv.nvmlDeviceGetPowerUsage(h) / 1000.0)
                                for h in self._handles) if x is not None]
            if utils:
                self.gpu_util.append(sum(utils) / len(utils))
            if mems:
                self.gpu_mem.append(sum(mems))
            if pows:
                self.gpu_power.append(sum(pows))

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            _safe(self._sample)

    def start(self) -> None:
        self._init_nvml()
        if psutil is not None:
            self._proc = psutil.Process(os.getpid())
            psutil.cpu_percent(interval=None)      # prime; first reading is meaningless
        self.start_time = time.time()
        _safe(lambda: self._sample(cpu=False))
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2 + 1)
        _safe(self._sample)
        self.end_time = time.time()
        if self._nvml is not None:
            _safe(self._nvml.nvmlShutdown)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _cfg_get(cfg: Any, section: str, name: str, default: Any = None) -> Any:
    """getattr chain that tolerates missing sections and OmegaConf MISSING values."""
    try:
        val = getattr(getattr(cfg, section), name)
        return default if val is None else val
    except Exception:
        return default


def _mean(xs: List[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _perplexity(loss: Optional[float]) -> Optional[float]:
    if loss is None:
        return None
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def _extract_losses(trainer: Any) -> Tuple[Optional[float], Optional[float]]:
    """Latest train loss and eval loss from trainer.state.log_history."""
    logs = getattr(getattr(trainer, "state", None), "log_history", None) or []
    loss = eval_loss = None
    for entry in reversed(logs):
        if loss is None and "loss" in entry:
            loss = float(entry["loss"])
        if eval_loss is None and "eval_loss" in entry:
            eval_loss = float(entry["eval_loss"])
    if loss is None:                                   # e.g. only the final summary row exists
        for entry in reversed(logs):
            if "train_loss" in entry:
                loss = float(entry["train_loss"])
                break
    return loss, eval_loss


def _count_params(model: Any) -> int:
    if model is None:
        return 0
    if hasattr(model, "num_parameters"):
        return int(model.num_parameters())
    if hasattr(model, "parameters"):
        return int(sum(p.numel() for p in model.parameters()))
    return 0


def _gpu_count(handles: list) -> int:
    if handles:
        return len(handles)
    try:
        import torch
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    if seconds < 3600:
        return f"{seconds / 60:.1f} minutes"
    return f"{seconds / 3600:.2f} hours"


_FLOPS_LEVELS = [
    (1e15, 1, "Very light", "a quick classroom demo on your laptop or desktop machine."),
    (1e17, 2, "Light", "a small fine-tuning run on non-production grade hardware (hardware References: NVIDIA RTX 3060/4060/4070; AMD RX 7600/7800 XT)"),
    (1e19, 3, "Moderate", "a substantial fine-tuning project on development grade hardware (hardware References: NVIDIA L4, L40, L40S; A10; AMD MI210)"),
    (1e21, 4, "Heavy", "a large research-lab job (hardware References: NVIDIA H100; AMD MI300X."),
    (float("inf"), 5, "Extreme", "a frontier-model scale training job (hardware References: DGX/HGX systems; NVIDIA GB200 NVL72; TPU pods; large AMD Instinct clusters "),
]


def explain_flops(flops: Optional[float]) -> Tuple[Optional[int], Optional[str]]:
    """Return (1-5 level, plain-language explanation) for a FLOP count."""
    if flops is None or flops <= 0:
        return None, None
    for limit, level, label, desc in _FLOPS_LEVELS:
        if flops < limit:
            break
    a100_seconds = flops / A100_EFFECTIVE_FLOPS
    text = (
        f"Level {level} of 5 ({label}), comparable to {desc}. A FLOP is one arithmetic "
        f"operation (an add or multiply) on a decimal number; this job performed about "
        f"{flops:.1e} of them. That is roughly {_fmt_duration(a100_seconds)} of work for one "
        f"NVIDIA A100 GPU at typical efficiency, or about "
        f"{100 * flops / GPT3_TRAIN_FLOPS:.2g}% of the compute used to train GPT-3."
    )
    return level, text


# --------------------------------------------------------------------------
# Tracker
# --------------------------------------------------------------------------
class MetricsTracker:
    def __init__(self, cfg: Any = None, interval: float = 1.0,
                 gpu_watts: float = DEFAULT_GPU_WATTS,
                 cpu_watts: float = DEFAULT_CPU_WATTS,
                 carbon_intensity: float = DEFAULT_CARBON_INTENSITY):
        self.cfg = cfg
        self.gpu_watts = gpu_watts
        self.cpu_watts = cpu_watts
        self.carbon_intensity = carbon_intensity
        self.monitor = ResourceMonitor(interval)
        self._counted_tokens = 0

    def start(self) -> None:
        self.monitor.start()

    def add_tokens(self, n: int) -> None:
        """Custom loops: call once per batch with the number of tokens processed."""
        self._counted_tokens += int(n)

    def _tokens(self, trainer: Any) -> Tuple[int, str]:
        if self._counted_tokens:
            return self._counted_tokens, "counted by tracker.add_tokens()"
        state = getattr(trainer, "state", None)
        seen = getattr(state, "num_input_tokens_seen", 0) or 0
        if seen:
            return int(seen), "reported by Trainer"
        # Fallback: optimizer steps x effective batch x max_length. Counts padding
        # tokens, so treat it as an upper bound.
        args = getattr(trainer, "args", None)
        steps = getattr(state, "global_step", 0) or 0
        per_device = getattr(args, "per_device_train_batch_size", 1) or 1
        accum = getattr(args, "gradient_accumulation_steps", 1) or 1
        world = getattr(args, "world_size", 1) or 1
        max_len = _cfg_get(self.cfg, "training", "max_length", 1024)
        return int(steps * per_device * accum * world * max_len), \
            "estimated from steps x batch x max_length (includes padding)"

    def stop(self, trainer: Any = None, model: Any = None,
             loss: Optional[float] = None, eval_loss: Optional[float] = None
             ) -> TrainingMetrics:
        mon = self.monitor
        mon.stop()
        m = TrainingMetrics()

        # Time
        m.start_time, m.end_time = mon.start_time, mon.end_time
        m.elapsed_time = (mon.end_time - mon.start_time) if mon.start_time else None

        # Memory
        if mon.rss:
            m.max_memory_bytes = max(mon.rss)
            m.min_memory_bytes = min(mon.rss)
            m.median_memory_bytes = int(statistics.median(mon.rss))
            m.total_memory_bytes = int(psutil.virtual_memory().total)
            m.memory_used_percent = 100.0 * m.max_memory_bytes / m.total_memory_bytes

        # Utilization
        m.avg_cpu_util_percent = _mean(mon.cpu)
        m.avg_gpu_util_percent = _mean(mon.gpu_util)

        # Tokens and budget
        m.token_budget = int(_cfg_get(self.cfg, "training", "token_budget", DEFAULT_TOKEN_BUDGET))
        m.tokens_processed, m.tokens_source = self._tokens(trainer)

        # FLOPs: ~6 * parameters * tokens (2N forward + 4N backward per token)
        model = model or getattr(trainer, "model", None)
        n_params = _safe(lambda: _count_params(model), 0)
        if n_params and m.tokens_processed:
            m.flops_estimate = 6.0 * n_params * m.tokens_processed
            m.flops_level, m.flops_explanation = explain_flops(m.flops_estimate)

        # Energy and carbon
        if m.elapsed_time is not None:
            n_gpu = _gpu_count(mon._handles)
            measured = _mean(mon.gpu_power)
            if measured is not None:
                gpu_w, src = measured, "measured"
            else:
                gpu_w, src = n_gpu * self.gpu_watts, "assumed"
            m.energy_kwh = (gpu_w + self.cpu_watts) * (m.elapsed_time / 3600.0) / 1000.0
            m.carbon_kg_estimate = m.energy_kwh * self.carbon_intensity
            gpu_note = (f"{n_gpu} GPU(s) at {gpu_w:.0f} W ({src})" if n_gpu
                        else "no GPU detected")
            m.carbon_explanation = (
                f"About {m.carbon_kg_estimate / KG_CO2_PER_KM_DRIVEN:.2f} km of driving in an "
                f"average gasoline car. Inputs: {gpu_note}; {self.cpu_watts:.0f} W CPU (assumed); "
                f"grid intensity {self.carbon_intensity} kg CO2e/kWh."
            )

        # Performance
        t_loss, t_eval = _extract_losses(trainer)
        m.loss = loss if loss is not None else t_loss
        m.eval_loss = eval_loss if eval_loss is not None else t_eval
        m.perplexity = _perplexity(m.loss)
        m.eval_perplexity = _perplexity(m.eval_loss)

        if self.cfg is not None:
            try:
                self.cfg.metrics = m
            except Exception:
                pass
        return m


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _na(v: Any, fmt: str = "{}", unit: str = "") -> str:
    return "N/A" if v is None else fmt.format(v) + unit


def _bytes(n: Optional[int]) -> str:
    if n is None:
        return "N/A"
    return f"{n:,} bytes ({n / 1024 ** 3:.2f} GB)"


def _report_sections(m: TrainingMetrics):
    econ = [
        ("Training time", _na(m.elapsed_time, "{:.1f}", " s")),
        ("Maximum memory used", _bytes(m.max_memory_bytes)),
        ("Minimum memory used", _bytes(m.min_memory_bytes)),
        ("Median memory used", _bytes(m.median_memory_bytes)),
        ("Peak memory as % of total", _na(m.memory_used_percent, "{:.1f}", "%")),
        ("Average CPU utilization", _na(m.avg_cpu_util_percent, "{:.1f}", "%")),
        ("Average GPU utilization", _na(m.avg_gpu_util_percent, "{:.1f}", "%")),
        ("Estimated FLOPs", "N/A" if m.flops_estimate is None
            else f"{m.flops_estimate:.2e} (level {m.flops_level} of 5)"),
        ("Estimated carbon footprint", _na(m.carbon_kg_estimate, "{:.4f}", " kg CO2e")),
        ("Token budget used", _na(m.token_budget_used_percent, "{:.1f}", "%")),
        ("Tokens processed", f"{m.tokens_processed:,}"),
        ("Tokens remaining", f"{m.tokens_remaining:,}"
            + (" (OVER BUDGET)" if m.over_budget else "")),
    ]
    perf = [
        ("Loss", _na(m.loss, "{:.4f}")),
        ("Perplexity", _na(m.perplexity, "{:.2f}")),
    ]
    if m.eval_loss is not None:
        perf += [("Eval loss", f"{m.eval_loss:.4f}"),
                 ("Eval perplexity", _na(m.eval_perplexity, "{:.2f}"))]
    notes = []
    if m.flops_explanation:
        notes.append(("What the FLOPs mean", m.flops_explanation))
    if m.carbon_explanation:
        notes.append(("What the carbon estimate means", m.carbon_explanation))
    if m.tokens_source:
        notes.append(("Token count source", f"{m.tokens_source}. Budget: {m.token_budget:,} tokens."))
    return econ, perf, notes


def format_report(m: TrainingMetrics, fmt: str = "text",
                  loss_curve: Optional[str] = None) -> str:
    """Render metrics as 'text' (console) or 'markdown' (file)."""
    econ, perf, notes = _report_sections(m)
    e_title, p_title = "LLM Training Job Economic Metrics", "LLM Training Performance"
    if fmt == "markdown":
        out = ["# Training Job Report", "", f"## {e_title}", "",
               "| Metric | Value |", "|---|---|"]
        out += [f"| {k} | {v} |" for k, v in econ]
        out += ["", f"## {p_title}", "", "| Metric | Value |", "|---|---|"]
        out += [f"| {k} | {v} |" for k, v in perf]
        if loss_curve:
            out += ["", f"![Loss curve]({loss_curve})"]
        if notes:
            out += ["", "## Notes", ""]
            out += [f"- **{k}:** {v}" for k, v in notes]
        return "\n".join(out) + "\n"

    width = max(len(k) for k, _ in econ + perf) + 2
    out = ["=" * 60, "TRAINING JOB REPORT", "=" * 60, "", e_title, "-" * len(e_title)]
    out += [f"{k + ':':<{width}}{v}" for k, v in econ]
    out += ["", p_title, "-" * len(p_title)]
    out += [f"{k + ':':<{width}}{v}" for k, v in perf]
    if notes:
        out += ["", "Notes", "-----"]
        out += [f"* {k}: {v}" for k, v in notes]
    return "\n".join(out) + "\n"


def plot_loss_curve(trainer: Any, path: str) -> Optional[str]:
    """Save a loss curve PNG from trainer.state.log_history. Needs matplotlib."""
    logs = getattr(getattr(trainer, "state", None), "log_history", None) or []
    train = [(e["step"], e["loss"]) for e in logs if "loss" in e and "step" in e]
    evals = [(e["step"], e["eval_loss"]) for e in logs if "eval_loss" in e and "step" in e]
    if not (train or evals):
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    fig, ax = plt.subplots(figsize=(6, 4))
    if train:
        ax.plot(*zip(*train), label="train loss")
    if evals:
        ax.plot(*zip(*evals), marker="o", label="eval loss")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_report(m: TrainingMetrics, output_dir: str, trainer: Any = None) -> Dict[str, Path]:
    """Write training_metrics.json, training_report.md, and (if possible) loss_curve.png."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    curve = plot_loss_curve(trainer, str(out / "loss_curve.png")) if trainer is not None else None
    if curve:
        paths["loss_curve"] = Path(curve)

    paths["json"] = out / "training_metrics.json"
    paths["json"].write_text(json.dumps(m.to_grouped_dict(), indent=2))
    paths["markdown"] = out / "training_report.md"
    paths["markdown"].write_text(
        format_report(m, "markdown", loss_curve="loss_curve.png" if curve else None))
    return paths
