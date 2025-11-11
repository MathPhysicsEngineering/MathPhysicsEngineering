"""Neural network policy for trade decision making."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch import nn

from ..config.schemas import ResearchArtifact, TrendingSymbol

LOGGER = logging.getLogger(__name__)


@dataclass
class Experience:
    features: np.ndarray
    reward: float


class FeatureVectorBuilder:
    """Transforms market and research data into model-friendly features."""

    FEATURE_SIZE = 16

    SENTIMENT_MAP = {
        "positive": 1.0,
        "neutral": 0.0,
        "negative": -1.0,
    }

    def build(self, trending: TrendingSymbol, research: Optional[ResearchArtifact]) -> np.ndarray:
        vec = np.zeros(self.FEATURE_SIZE, dtype=np.float32)
        vec[0] = np.tanh(trending.premarket_change_percent / 20)
        vec[1] = np.log1p(max(trending.premarket_volume, 0)) / 15
        vec[2] = np.log1p(max(trending.last_price, 0)) / 10
        if research:
            vec[3] = self.SENTIMENT_MAP.get(research.sentiment.value, 0.0)
            vec[4] = float(research.analyst_price_target or 0.0) / 1000
            vec[5] = float(research.cashflow_strength or 0.0) / 1e9
            vec[6] = len(research.risk_flags) / 10
            vec[7] = 1.0 if research.analyst_consensus in {"upgrade", "buy"} else 0.0
            vec[8] = 1.0 if research.analyst_consensus in {"downgrade", "sell"} else 0.0
            vec[9] = 1.0 if research.summary else 0.0
        vec[10] = np.tanh((trending.timestamp.timestamp() % 86400) / 86400)
        vec[11] = 1.0 if trending.reason and "ibkr" in trending.reason else 0.0
        vec[12] = 1.0 if trending.reason and "yahoo" in trending.reason else 0.0
        vec[13] = np.clip(trending.premarket_change_percent / 100, -1.0, 1.0)
        vec[14] = np.clip(trending.premarket_volume / 1e7, 0.0, 1.0)
        vec[15] = 1.0
        return vec


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralPolicy:
    """Trainable neural network policy driven by virtual trading rewards."""

    def __init__(
        self,
        feature_builder: FeatureVectorBuilder,
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
        batch_size: int = 64,
        gamma: float = 0.99,
    ) -> None:
        self._builder = feature_builder
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._network = PolicyNetwork(feature_builder.FEATURE_SIZE).to(self._device)
        self._optimizer = torch.optim.Adam(self._network.parameters(), lr=learning_rate)
        self._batch_size = batch_size
        self._gamma = gamma
        self._experience: List[Experience] = []

    def predict(self, trending: TrendingSymbol, research: Optional[ResearchArtifact]) -> float:
        features = self._builder.build(trending, research)
        tensor = torch.from_numpy(features).unsqueeze(0).to(self._device)
        with torch.no_grad():
            score = self._network(tensor).item()
        return float(score)

    def record(self, features: np.ndarray, reward: float) -> None:
        self._experience.append(Experience(features=features.astype(np.float32), reward=reward))

    def train(self) -> float:
        if len(self._experience) < self._batch_size:
            return 0.0
        batch = self._sample_batch(self._batch_size)
        features = torch.from_numpy(np.stack([exp.features for exp in batch])).to(self._device)
        rewards = torch.tensor([exp.reward for exp in batch], dtype=torch.float32).to(self._device)
        returns = self._discount(rewards)

        preds = self._network(features).squeeze()
        loss = nn.functional.mse_loss(preds, returns)
        self._optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self._network.parameters(), max_norm=1.0)
        self._optimizer.step()
        return float(loss.item())

    def _sample_batch(self, size: int) -> Sequence[Experience]:
        indices = np.random.choice(len(self._experience), size=size, replace=False)
        return [self._experience[i] for i in indices]

    def _discount(self, rewards: torch.Tensor) -> torch.Tensor:
        discounted = torch.zeros_like(rewards)
        running = 0.0
        for idx in reversed(range(len(rewards))):
            running = rewards[idx] + self._gamma * running
            discounted[idx] = running
        min_val = discounted.min().item()
        max_val = discounted.max().item()
        if max_val - min_val > 1e-6:
            discounted = (discounted - min_val) / (max_val - min_val)
        return discounted

    def save_state(self) -> dict:
        return {
            "model_state": self._network.state_dict(),
            "optimizer_state": self._optimizer.state_dict(),
        }

    def load_state(self, state: dict) -> None:
        self._network.load_state_dict(state["model_state"])
        self._optimizer.load_state_dict(state["optimizer_state"])


