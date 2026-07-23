import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple

# DO NOT CHANGE SEED (used for deterministic init)
np.random.seed(42)

def _to_tensor(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)

class Layer:
    def set_input_shape(self, shape: Tuple[int, ...]):
        self.input_shape = shape

    def layer_name(self) -> str:
        return self.__class__.__name__

    def parameters(self) -> int:
        return 0

    def forward_pass(self, X, training: bool = True):
        raise NotImplementedError

    def backward_pass(self, accum_grad):
        raise NotImplementedError

    def output_shape(self) -> Tuple[int, ...]:
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, n_units: int, input_shape: Tuple[int, ...] | None = None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None  # torch.Tensor with requires_grad=True
        self.b = None  # torch.Tensor with requires_grad=True
        self._opt = None

    def initialize(self, optimizer) -> None:
        """Initialize weights with a uniform distribution and biases with zeros.
        Hint: use numpy to create deterministic arrays, then convert to torch tensors.
        Also store the provided optimizer for updates.
        """

        limit = 1 / np.sqrt(self.input_shape[0])
        W_np = np.random.uniform(
            -limit, limit, size=(self.input_shape[0], self.n_units)
        )
        self.W = torch.tensor(W_np, dtype=torch.float64, requires_grad=True)

        self.b = torch.zeros(size=(self.n_units,), dtype=torch.float64, requires_grad=True)
        self._opt = optimizer

    def number_of_parameters(self) -> int:
        """Return total number of trainable parameters in W and b."""
        return self.W.numel() + self.b.numel()

    def forward_pass(self, X, training: bool = True):
        """Use torch.nn.functional.linear for the forward pass (X @ W + b)."""
        self.layer_input = torch.tensor(X, dtype=torch.float64)
        self.output =  F.linear(self.layer_input, self.W.T, self.b)
        return self.output

    def backward_pass(self, accum_grad):
        """Use torch.autograd.grad to get dL/dW, dL/db given upstream gradient.
        Then update params with the provided optimizer and return grad w.r.t. input.
        Hint: grad_input can be computed with a matmul against W.T.
        """
        accum_grad = torch.as_tensor(np.asarray(accum_grad, dtype=str).astype(np.float64), dtype=torch.float64)
        W = self.W
        grad_W, grad_b = torch.autograd.grad(self.output, (self.W, self.b), grad_outputs=accum_grad)
        if self.trainable:
            self.W = self._opt.update(self.W, grad_W)
            self.b = self._opt.update(self.b, grad_b)
        return accum_grad @ W.T