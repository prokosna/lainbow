"""Interfaces implemented by engine models."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class EmbedResult:
    """An L2-normalized embedding of length ``MODELS[id].dimension`` plus optional details."""

    embedding: npt.NDArray[np.floating[Any]]
    details: Mapping[str, Any] | None = None


class AudioModel(Protocol):
    """A model that turns audio clips into one embedding."""

    id: str

    def load(self, device: str) -> None:
        """Load weights onto the device."""
        ...

    def unload(self) -> None:
        """Release the weights (and cached device memory); ``load`` can be called again."""
        ...

    def embed(self, clips: Sequence[npt.NDArray[np.float32]]) -> EmbedResult:
        """Embed the clips of one track."""
        ...


class TextModel(Protocol):
    """A model that can also embed text into the same space."""

    def embed_text(self, text: str) -> npt.NDArray[np.floating[Any]]:
        """Embed a text query."""
        ...
