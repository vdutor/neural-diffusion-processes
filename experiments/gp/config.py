from typing import Mapping
import dataclasses


@dataclasses.dataclass
class DataConfig:
    kernel: str = "rbf"
    num_samples: int = 10_000
    num_points: int = 100
    hyperparameters: Mapping[str, float] = dataclasses.field(
        default_factory=lambda: {
            "variance": 1.0,
            "lengthscale": 0.2,
        }
    )


@dataclasses.dataclass
class OptimizationConfig:
    batch_size: int = 16
    num_steps: int = 100_000


@dataclasses.dataclass
class NetworkConfig:
    num_bidim_attention_layers: int = 2
    hidden_dim: int = 16
    num_heads: int = 4


@dataclasses.dataclass
class Config:
    seed: int = 42
    data: DataConfig = DataConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    network: NetworkConfig = NetworkConfig()
