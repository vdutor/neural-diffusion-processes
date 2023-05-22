from dataclasses import dataclass



@dataclass
class DiffusionConfig:
    schedule: str = "cosine"
    beta_start: float = 3e-4
    beta_end: float = 0.5
    timesteps: int = 1_000


@dataclass
class OptimizerConfig:
    num_warmup_epochs: int = 10
    num_decay_epochs: int = 200
    init_lr: float = 2e-5
    peak_lr: float = 1e-3
    end_lr: float = 1e-5
    ema_rate: float = 0.995  # 0.999

@dataclass
class NetworkConfig:
    n_layers: int = 2
    hidden_dim: int = 64
    num_heads: int = 8


@dataclass
class Config:
    seed: int = 42
    batch_size: int = 32
    epochs: int = 500
    samples_per_epoch: int = int(2**14)
    loss_type: str = "l1"
    dataset: str = "se"
    eval_every: int = 2000
    log_every: int = 200
    network: NetworkConfig = NetworkConfig()
    schedule: DiffusionConfig = DiffusionConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    optimizer: OptimizerConfig = OptimizerConfig()

    @property
    def steps_per_epoch(self) -> int:
        return self.samples_per_epoch // self.batch_size

    @property
    def total_steps(self) -> int:
        return self.steps_per_epoch * self.epochs


