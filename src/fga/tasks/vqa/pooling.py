"""Compact Bilinear Pooling, used to fuse the attended modalities.

A full bilinear product of two `d`-dimensional vectors has `d^2` terms, which is
hopeless at `d = 512`. Compact Bilinear Pooling (Gao et al., 2016; Fukui et al.,
2016) approximates it by sketching each vector into `output_dim` bins with the
Count Sketch, then exploiting the fact that the sketch of an outer product is the
convolution of the sketches — and convolution is a product in Fourier space:

    CBP(x, y) = ifft( fft(sketch(x)) * fft(sketch(y)) )

so the cost is `O(d + output_dim log output_dim)` rather than `O(d^2)`.

The sketch's random projections are fixed at construction and registered as
buffers, so they are saved with the model: they are part of the function, and
resampling them on reload would silently change what the layer computes.
"""

from typing import Optional

import torch
import torch.nn as nn

__all__ = ["CompactBilinearPooling", "count_sketch"]


def count_sketch(x: torch.Tensor, hash_index: torch.Tensor, hash_sign: torch.Tensor, output_dim: int):
    """Project `x` into `output_dim` bins, summing each feature into one bin.

    Args:
        x: `(batch, dim)`.
        hash_index: `(dim,)` target bin per input feature.
        hash_sign: `(dim,)` random +-1 per input feature, which makes the sketch
            unbiased.
        output_dim: number of bins.
    """
    sketch = x.new_zeros(x.size(0), output_dim)
    return sketch.index_add_(1, hash_index, x * hash_sign.to(x.dtype))


class CompactBilinearPooling(nn.Module):
    """Approximate the outer product of two vectors in `output_dim` dimensions.

    Args:
        dim_x / dim_y: input dimensions.
        output_dim: sketch size. Larger is a closer approximation; the VQA models
            that popularised this used 16000.
        sum_pool: unused placeholder kept for signature compatibility with other
            implementations; inputs here are already pooled vectors.
        seed: fixes the random projections.

    Shape:
        - Input: `(batch, dim_x)` and `(batch, dim_y)`
        - Output: `(batch, output_dim)`
    """

    def __init__(
        self,
        dim_x: int,
        dim_y: int,
        output_dim: int = 16000,
        sum_pool: bool = False,
        seed: Optional[int] = 0,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.sum_pool = sum_pool

        generator = torch.Generator().manual_seed(seed) if seed is not None else None

        def sketch_parameters(dim: int):
            index = torch.randint(0, output_dim, (dim,), generator=generator)
            sign = torch.randint(0, 2, (dim,), generator=generator) * 2 - 1
            return index, sign.float()

        x_index, x_sign = sketch_parameters(dim_x)
        y_index, y_sign = sketch_parameters(dim_y)

        # Buffers, not parameters: fixed random projections that define the
        # function and must travel with the checkpoint.
        self.register_buffer("x_index", x_index)
        self.register_buffer("x_sign", x_sign)
        self.register_buffer("y_index", y_index)
        self.register_buffer("y_sign", y_sign)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Computed in float32 regardless of autocast: torch.fft has no bf16
        # kernels, and the sketch is a rounding-sensitive sum of many terms.
        # The cost is negligible next to the attention.
        x, y = x.float(), y.float()
        sketch_x = count_sketch(x, self.x_index, self.x_sign, self.output_dim)
        sketch_y = count_sketch(y, self.y_index, self.y_sign, self.output_dim)

        # Convolution of the sketches, done as a product of their spectra.
        fft_x = torch.fft.rfft(sketch_x, n=self.output_dim, dim=-1)
        fft_y = torch.fft.rfft(sketch_y, n=self.output_dim, dim=-1)
        return torch.fft.irfft(fft_x * fft_y, n=self.output_dim, dim=-1)


def signed_sqrt(x: torch.Tensor) -> torch.Tensor:
    """The signed square root the VQA models apply after pooling.

    The pooled features are heavy-tailed; this compresses the magnitudes while
    keeping the sign, and is normally followed by L2 normalization.
    """
    return torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-12)
