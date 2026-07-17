import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Which block of BLOCK_SIZE elements is this program instance responsible for?
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # The element indices this program will process.
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Guard against reading/writing past the end of the tensors.
    mask = offsets < n_elements

    # Load, add, store.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Allocate the output with the same shape/dtype/device as x.
    output = torch.empty_like(x)
    n_elements = output.numel()

    # One program instance per BLOCK_SIZE chunk of elements.
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch the kernel. Tensors are auto-converted to their data pointers.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return output
