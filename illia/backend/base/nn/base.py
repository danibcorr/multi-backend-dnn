# Standard libraries
from abc import ABC, abstractmethod


class LinearBase(ABC):
    """Interface for a fully-connected layer."""

    @abstractmethod
    def forward(self, x):
        """Run the forward pass."""


class Conv2dBase(ABC):
    """Interface for a 2-D convolution layer."""

    @abstractmethod
    def forward(self, x):
        """Run the forward pass."""
