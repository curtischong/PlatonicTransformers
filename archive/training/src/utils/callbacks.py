import time
import torch
import logging
from typing import Any, Optional


log = logging.getLogger(__name__)

from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning import Trainer
from schedulefree import AdamWScheduleFree



class StepProgressCallback(Callback):
    """Print text progress bar lines every N steps with epoch ETA."""

    def __init__(self, every_n_steps: int):
        super().__init__()
        self.every_n_steps = max(1, int(every_n_steps))
        self.bar_width = 30
        self._epoch_start_time = None

    @staticmethod
    def _format_duration(seconds):
        if seconds is None or seconds < 0:
            return "??:??:??"
        total = int(seconds)
        hours, rem = divmod(total, 3600)
        minutes, secs = divmod(rem, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def _extract_loss(outputs):
        if torch.is_tensor(outputs):
            return float(outputs.detach().item())
        if isinstance(outputs, dict):
            loss = outputs.get("loss")
            if torch.is_tensor(loss):
                return float(loss.detach().item())
            if isinstance(loss, (int, float)):
                return float(loss)
        return None

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            self._epoch_start_time = time.monotonic()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if not trainer.is_global_zero:
            return

        global_step = int(trainer.global_step)
        epoch_step = int(batch_idx) + 1

        try:
            total_batches = int(trainer.num_training_batches)
        except (TypeError, ValueError):
            total_batches = None

        is_last_batch = total_batches is not None and epoch_step >= total_batches
        if global_step == 0 or (global_step % self.every_n_steps != 0 and not is_last_batch):
            return

        loss_val = self._extract_loss(outputs)

        elapsed = None
        if self._epoch_start_time is not None:
            elapsed = max(0.0, time.monotonic() - self._epoch_start_time)

        if total_batches is None or total_batches <= 0:
            msg = f"Epoch {trainer.current_epoch} | global_step={global_step}"
            if loss_val is not None:
                msg += f" | train_loss={loss_val:.6f}"
            log.info(msg)
            return

        progress = min(max(epoch_step / total_batches, 0.0), 1.0)
        filled = int(self.bar_width * progress)
        bar = "#" * filled + "-" * (self.bar_width - filled)

        it_per_sec = None
        eta = None
        if elapsed is not None and elapsed > 0:
            it_per_sec = epoch_step / elapsed
            if it_per_sec > 0:
                eta = (total_batches - epoch_step) / it_per_sec

        msg = (
            f"Epoch {trainer.current_epoch} [{bar}] "
            f"{epoch_step}/{total_batches} ({progress * 100:5.1f}%)"
        )

        if it_per_sec is not None:
            msg += f" | {it_per_sec:6.2f} it/s"
        else:
            msg += " |   n/a it/s"

        msg += f" | elapsed {self._format_duration(elapsed)}"
        msg += f" | eta {self._format_duration(eta)}"

        if loss_val is not None:
            msg += f" | train_loss={loss_val:.6f}"

        log.info(msg)







class SchedulerFreeModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        opts = trainer.optimizers
        if not isinstance(opts, list):
            opts = [opts]

        # save current train/eval state so that we can restore them
        states = []
        for opt in opts:
            if isinstance(opt, AdamWScheduleFree):
                # probe optimizer state by checking the train state of the first parameter group
                is_train_mode = opt.param_groups[0]["train_mode"]
                if is_train_mode:
                    opt.eval()
            else:
                is_train_mode = None
            states.append(is_train_mode)

        super()._save_checkpoint(trainer, filepath)

        # restore previous state
        for (opt, wasintrainmode) in zip(opts, states):
            if isinstance(opt, AdamWScheduleFree):
                if wasintrainmode:
                    opt.train()


class EMACallback(Callback):
    """Exponential moving average of trainable weights, swapped in for val/test.

    ema_theta <- decay * ema_theta + (1 - decay) * theta after every train batch.
    During warmup_steps, effective decay = min(decay, 1 - 1/(step+1)) so early
    high-LR weights don't dominate. State kept fp32 regardless of training dtype.
    """

    def __init__(self, decay: float = 0.9999, warmup_steps: int = 2000) -> None:
        super().__init__()
        self.decay = decay
        self.warmup_steps = warmup_steps
        self._ema_state: Optional[dict] = None
        self._saved_online_state: Optional[dict] = None

    def _iter_trainable(self, pl_module):
        for name, p in pl_module.named_parameters():
            if p.requires_grad:
                yield name, p

    def _init_ema(self, pl_module) -> None:
        self._ema_state = {
            name: p.detach().clone().float()
            for name, p in self._iter_trainable(pl_module)
        }

    def _effective_decay(self, global_step: int) -> float:
        if self.warmup_steps > 0 and global_step < self.warmup_steps:
            return min(self.decay, 1.0 - 1.0 / (global_step + 1))
        return self.decay

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if self._ema_state is None:
            self._init_ema(pl_module)
        d = self._effective_decay(trainer.global_step)
        with torch.no_grad():
            for name, p in self._iter_trainable(pl_module):
                ema_p = self._ema_state.get(name)
                if ema_p is None:
                    self._ema_state[name] = p.detach().clone().float()
                    continue
                if ema_p.device != p.device:
                    ema_p = ema_p.to(p.device)
                    self._ema_state[name] = ema_p
                ema_p.mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    def _swap_in(self, pl_module) -> None:
        if self._ema_state is None:
            return
        self._saved_online_state = {}
        with torch.no_grad():
            for name, p in self._iter_trainable(pl_module):
                self._saved_online_state[name] = p.detach().clone()
                ema_p = self._ema_state.get(name)
                if ema_p is not None:
                    p.data.copy_(ema_p.to(p.dtype).to(p.device))

    def _swap_out(self, pl_module) -> None:
        if self._saved_online_state is None:
            return
        with torch.no_grad():
            for name, p in self._iter_trainable(pl_module):
                saved = self._saved_online_state.get(name)
                if saved is not None:
                    p.data.copy_(saved)
        self._saved_online_state = None

    def on_validation_start(self, trainer, pl_module) -> None:
        self._swap_in(pl_module)

    def on_validation_end(self, trainer, pl_module) -> None:
        self._swap_out(pl_module)

    def on_test_start(self, trainer, pl_module) -> None:
        self._swap_in(pl_module)

    def on_test_end(self, trainer, pl_module) -> None:
        self._swap_out(pl_module)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "ema_state": self._ema_state,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.decay = float(state_dict.get("decay", self.decay))
        self.warmup_steps = int(state_dict.get("warmup_steps", self.warmup_steps))
        self._ema_state = state_dict.get("ema_state", None)