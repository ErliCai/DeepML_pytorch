# [968. Triton: Vector Addition Kernel](https://www.deep-ml.com/problems/968)

## Problem Description

The Canonical First Two-Input Kernel
Write a Triton kernel add_kernel and a Python wrapper add(x, y) that computes element-wise x + y on the GPU.

Required signature for the kernel:

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
Required Python wrapper:

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor
The wrapper should:

Allocate an output tensor of the same shape and dtype as x.
Launch add_kernel with BLOCK_SIZE=1024 on a 1-D grid sized via triton.cdiv.
Return the output.
Inputs may have any length, not just multiples of BLOCK_SIZE - use a mask.

Hints
Inside the kernel: get pid = tl.program_id(0), then build offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).
tl.load(ptr + offsets, mask=mask) reads with bounds-checking; tl.store(ptr + offsets, value, mask=mask) writes with bounds-checking.
triton.cdiv(n, BLOCK_SIZE) computes the ceiling-division for the grid size.
Don't forget the BLOCK_SIZE: tl.constexpr annotation - it tells Triton this is a compile-time constant.

## Example

Input:
x = torch.tensor([1., 2., 3.]), y = torch.tensor([10., 20., 30.])
Output:
tensor([11., 22., 33.])

## Explanation

TODO

## Solution

TODO
