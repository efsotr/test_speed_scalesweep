# SPDX-License-Identifier: Apache-2.0
# Reference-compatible wrapper for kernel_ScaleSweep_MSE.py.

import torch

try:
    from .custom import create_fp4_output_tensors
except ImportError:  # Allows running the benchmark script by path.
    from custom import create_fp4_output_tensors
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up

BLOCK_SIZE = 16
LOWER_BOUND = -3
UPPER_BOUND = 7


@triton.jit
def vllm_swizzled_scale_offsets(
    block_offsets,
    BLOCKS_PER_OUT: tl.constexpr,
    K_PAD: tl.constexpr,
):
    row = block_offsets // BLOCKS_PER_OUT
    col = block_offsets % BLOCKS_PER_OUT

    major_m = row >> 7
    row_in_tile = row & 127
    tile_m = row_in_tile >> 5
    inner_m = row_in_tile & 31

    major_k = col >> 2
    inner_k = col & 3

    return (
        major_m * (K_PAD * 128)
        + major_k * 512
        + inner_m * 16
        + tile_m * 4
        + inner_k
    )


@triton.jit
def _fp32x16_to_e2m1_u32x2(
    x0, x1, x2, x3,
    x4, x5, x6, x7,
    x8, x9, x10, x11,
    x12, x13, x14, x15,
):
    lo, hi = tl.inline_asm_elementwise(
        asm="""
        {
          .reg .b8 b0;
          .reg .b8 b1;
          .reg .b8 b2;
          .reg .b8 b3;
          .reg .b8 b4;
          .reg .b8 b5;
          .reg .b8 b6;
          .reg .b8 b7;

          cvt.rn.satfinite.e2m1x2.f32 b0,  $3,  $2;
          cvt.rn.satfinite.e2m1x2.f32 b1,  $5,  $4;
          cvt.rn.satfinite.e2m1x2.f32 b2,  $7,  $6;
          cvt.rn.satfinite.e2m1x2.f32 b3,  $9,  $8;
          cvt.rn.satfinite.e2m1x2.f32 b4,  $11, $10;
          cvt.rn.satfinite.e2m1x2.f32 b5,  $13, $12;
          cvt.rn.satfinite.e2m1x2.f32 b6,  $15, $14;
          cvt.rn.satfinite.e2m1x2.f32 b7,  $17, $16;

          mov.b32 $0, {b0, b1, b2, b3};
          mov.b32 $1, {b4, b5, b6, b7};
        }
        """,
        constraints="=r,=r,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f,f",
        args=[
            x0, x1, x2, x3,
            x4, x5, x6, x7,
            x8, x9, x10, x11,
            x12, x13, x14, x15,
        ],
        dtype=(tl.uint32, tl.uint32),
        is_pure=True,
        pack=1,
    )

    return lo, hi


@triton.jit
def _fp32_pair_e2m1_roundtrip_squared_error(x0, x1):
    squared_error = tl.inline_asm_elementwise(
        asm=r"""
        {
          .reg .b8  b;
          .reg .b32 h;
          .reg .b16 lo;
          .reg .b16 hi;
          .reg .f32 q0;
          .reg .f32 q1;
          .reg .f32 d0;
          .reg .f32 d1;

          cvt.rn.satfinite.e2m1x2.f32 b, $2, $1;
          cvt.rn.f16x2.e2m1x2 h, b;

          mov.b32 {lo, hi}, h;
          cvt.f32.f16 q0, lo;
          cvt.f32.f16 q1, hi;

          sub.rn.f32 d0, q0, $1;
          sub.rn.f32 d1, q1, $2;
          mul.rn.f32 d0, d0, d0;
          fma.rn.f32 $0, d1, d1, d0;
        }
        """,
        constraints="=f,f,f",
        args=[x0, x1],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    return squared_error


@triton.jit
def _fp32x16_e2m1_roundtrip_squared_error(
    x0, x1, x2, x3,
    x4, x5, x6, x7,
    x8, x9, x10, x11,
    x12, x13, x14, x15,
):
    s01 = _fp32_pair_e2m1_roundtrip_squared_error(x0, x1)
    s23 = _fp32_pair_e2m1_roundtrip_squared_error(x2, x3)
    s45 = _fp32_pair_e2m1_roundtrip_squared_error(x4, x5)
    s67 = _fp32_pair_e2m1_roundtrip_squared_error(x6, x7)
    s89 = _fp32_pair_e2m1_roundtrip_squared_error(x8, x9)
    sAB = _fp32_pair_e2m1_roundtrip_squared_error(x10, x11)
    sCD = _fp32_pair_e2m1_roundtrip_squared_error(x12, x13)
    sEF = _fp32_pair_e2m1_roundtrip_squared_error(x14, x15)

    return ((s01 + s23) + (s45 + s67)) + ((s89 + sAB) + (sCD + sEF))


@triton.jit
def _squared_error_16_cols_after_e2m1_roundtrip(
    v0, v1, v2, v3,
    v4, v5, v6, v7,
    v8, v9, v10, v11,
    v12, v13, v14, v15,
    inv_scale,
    scale,
):
    x0 = v0 * inv_scale
    x1 = v1 * inv_scale
    x2 = v2 * inv_scale
    x3 = v3 * inv_scale
    x4 = v4 * inv_scale
    x5 = v5 * inv_scale
    x6 = v6 * inv_scale
    x7 = v7 * inv_scale
    x8 = v8 * inv_scale
    x9 = v9 * inv_scale
    x10 = v10 * inv_scale
    x11 = v11 * inv_scale
    x12 = v12 * inv_scale
    x13 = v13 * inv_scale
    x14 = v14 * inv_scale
    x15 = v15 * inv_scale

    squared_error = _fp32x16_e2m1_roundtrip_squared_error(
        x0, x1, x2, x3,
        x4, x5, x6, x7,
        x8, x9, x10, x11,
        x12, x13, x14, x15,
    )
    return squared_error * (scale * scale)


@triton.jit
def _pack_final_code_16_cols(
    v0, v1, v2, v3,
    v4, v5, v6, v7,
    v8, v9, v10, v11,
    v12, v13, v14, v15,
    inv_scale,
):
    x0 = v0 * inv_scale
    x1 = v1 * inv_scale
    x2 = v2 * inv_scale
    x3 = v3 * inv_scale
    x4 = v4 * inv_scale
    x5 = v5 * inv_scale
    x6 = v6 * inv_scale
    x7 = v7 * inv_scale
    x8 = v8 * inv_scale
    x9 = v9 * inv_scale
    x10 = v10 * inv_scale
    x11 = v11 * inv_scale
    x12 = v12 * inv_scale
    x13 = v13 * inv_scale
    x14 = v14 * inv_scale
    x15 = v15 * inv_scale

    return _fp32x16_to_e2m1_u32x2(
        x0, x1, x2, x3,
        x4, x5, x6, x7,
        x8, x9, x10, x11,
        x12, x13, x14, x15,
    )


@triton.jit
def _max_abs_16(
    v0, v1, v2, v3,
    v4, v5, v6, v7,
    v8, v9, v10, v11,
    v12, v13, v14, v15,
):
    m0 = tl.maximum(tl.abs(v0), tl.abs(v1))
    m1 = tl.maximum(tl.abs(v2), tl.abs(v3))
    m2 = tl.maximum(tl.abs(v4), tl.abs(v5))
    m3 = tl.maximum(tl.abs(v6), tl.abs(v7))
    m4 = tl.maximum(tl.abs(v8), tl.abs(v9))
    m5 = tl.maximum(tl.abs(v10), tl.abs(v11))
    m6 = tl.maximum(tl.abs(v12), tl.abs(v13))
    m7 = tl.maximum(tl.abs(v14), tl.abs(v15))

    m01 = tl.maximum(m0, m1)
    m23 = tl.maximum(m2, m3)
    m45 = tl.maximum(m4, m5)
    m67 = tl.maximum(m6, m7)

    m0123 = tl.maximum(m01, m23)
    m4567 = tl.maximum(m45, m67)

    return tl.maximum(m0123, m4567)


@triton.jit
def _load_normalized_16_cols(
    ptr,
    block_offsets,
    block_mask,
    global_scale_inv,
):
    base_elem = block_offsets * 16

    v0 = tl.load(ptr + base_elem + 0, mask=block_mask, other=0.0)
    v1 = tl.load(ptr + base_elem + 1, mask=block_mask, other=0.0)
    v2 = tl.load(ptr + base_elem + 2, mask=block_mask, other=0.0)
    v3 = tl.load(ptr + base_elem + 3, mask=block_mask, other=0.0)
    v4 = tl.load(ptr + base_elem + 4, mask=block_mask, other=0.0)
    v5 = tl.load(ptr + base_elem + 5, mask=block_mask, other=0.0)
    v6 = tl.load(ptr + base_elem + 6, mask=block_mask, other=0.0)
    v7 = tl.load(ptr + base_elem + 7, mask=block_mask, other=0.0)
    v8 = tl.load(ptr + base_elem + 8, mask=block_mask, other=0.0)
    v9 = tl.load(ptr + base_elem + 9, mask=block_mask, other=0.0)
    v10 = tl.load(ptr + base_elem + 10, mask=block_mask, other=0.0)
    v11 = tl.load(ptr + base_elem + 11, mask=block_mask, other=0.0)
    v12 = tl.load(ptr + base_elem + 12, mask=block_mask, other=0.0)
    v13 = tl.load(ptr + base_elem + 13, mask=block_mask, other=0.0)
    v14 = tl.load(ptr + base_elem + 14, mask=block_mask, other=0.0)
    v15 = tl.load(ptr + base_elem + 15, mask=block_mask, other=0.0)

    v0 = v0.to(tl.float32) * global_scale_inv
    v1 = v1.to(tl.float32) * global_scale_inv
    v2 = v2.to(tl.float32) * global_scale_inv
    v3 = v3.to(tl.float32) * global_scale_inv
    v4 = v4.to(tl.float32) * global_scale_inv
    v5 = v5.to(tl.float32) * global_scale_inv
    v6 = v6.to(tl.float32) * global_scale_inv
    v7 = v7.to(tl.float32) * global_scale_inv
    v8 = v8.to(tl.float32) * global_scale_inv
    v9 = v9.to(tl.float32) * global_scale_inv
    v10 = v10.to(tl.float32) * global_scale_inv
    v11 = v11.to(tl.float32) * global_scale_inv
    v12 = v12.to(tl.float32) * global_scale_inv
    v13 = v13.to(tl.float32) * global_scale_inv
    v14 = v14.to(tl.float32) * global_scale_inv
    v15 = v15.to(tl.float32) * global_scale_inv

    return (
        v0, v1, v2, v3,
        v4, v5, v6, v7,
        v8, v9, v10, v11,
        v12, v13, v14, v15,
    )


SCALESWEEP_CONFIGS = [
    triton.Config({"BLOCKS_PER_PROGRAM": 32, "NUM_STAGES": 2}, num_warps=1),
    triton.Config({"BLOCKS_PER_PROGRAM": 64, "NUM_STAGES": 2}, num_warps=2),
    triton.Config({"BLOCKS_PER_PROGRAM": 128, "NUM_STAGES": 2}, num_warps=4),
    triton.Config({"BLOCKS_PER_PROGRAM": 256, "NUM_STAGES": 2}, num_warps=8),
    triton.Config({"BLOCKS_PER_PROGRAM": 512, "NUM_STAGES": 2}, num_warps=16),
    triton.Config({"BLOCKS_PER_PROGRAM": 1024, "NUM_STAGES": 2}, num_warps=32),
]

@triton.autotune(
    configs=SCALESWEEP_CONFIGS,
    key=["NUM_BLOCKS", "LOWER_BOUND", "NUM_CANDIDATES", "BLOCKS_PER_OUT", "K_PAD", "IS_SWIZZLE_SCALE"],
)
@triton.jit
def scalesweep_quantize_kernel(
    weight_ptr,
    scale_ptr,
    code_i32_ptr,
    global_scale_inv_ptr,
    NUM_BLOCKS: tl.constexpr,
    BLOCKS_PER_OUT: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    IS_SWIZZLE_SCALE: tl.constexpr,
    K_PAD: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    global_scale_inv = tl.load(global_scale_inv_ptr)

    pid = tl.program_id(0)
    block_start = pid * BLOCKS_PER_PROGRAM

    block_offsets = block_start + tl.arange(0, BLOCKS_PER_PROGRAM)
    block_mask = block_offsets < NUM_BLOCKS

    (
        v0, v1, v2, v3,
        v4, v5, v6, v7,
        v8, v9, v10, v11,
        v12, v13, v14, v15,
    ) = _load_normalized_16_cols(
        weight_ptr,
        block_offsets,
        block_mask,
        global_scale_inv,
    )

    abs_max = _max_abs_16(
        v0, v1, v2, v3,
        v4, v5, v6, v7,
        v8, v9, v10, v11,
        v12, v13, v14, v15,
    )

    base_scale = abs_max * (1.0 / 6.0)
    base_fp8 = base_scale.to(tl.float8e4nv)
    base_raw = base_fp8.to(tl.uint8, bitcast=True).to(tl.int32) - (
        base_fp8.to(tl.float32) > base_scale
    ).to(tl.int32)

    best_mse = tl.full((BLOCKS_PER_PROGRAM,), float("inf"), tl.float32)
    best_scale_fp8 = tl.full((BLOCKS_PER_PROGRAM,), 0, tl.float8e4nv)

    for i in tl.static_range(0, NUM_CANDIDATES):
        raw_i = tl.minimum(
            tl.maximum(base_raw + LOWER_BOUND + i, 1),
            126,
        ).to(tl.uint8)

        scale_fp8 = raw_i.to(tl.float8e4nv, bitcast=True)
        scale_i = scale_fp8.to(tl.float32)
        inv_scale_i = 1.0 / scale_i

        mse_i = _squared_error_16_cols_after_e2m1_roundtrip(
            v0, v1, v2, v3,
            v4, v5, v6, v7,
            v8, v9, v10, v11,
            v12, v13, v14, v15,
            inv_scale_i,
            scale_i,
        )

        better = mse_i < best_mse
        best_mse = tl.where(better, mse_i, best_mse)
        best_scale_fp8 = tl.where(better, scale_fp8, best_scale_fp8)

    scale_offsets = block_offsets
    if IS_SWIZZLE_SCALE:
        scale_offsets = vllm_swizzled_scale_offsets(
            block_offsets,
            BLOCKS_PER_OUT,
            K_PAD,
        )
    tl.store(scale_ptr + scale_offsets, best_scale_fp8, mask=block_mask)

    best_scale_inv = 1.0 / best_scale_fp8.to(tl.float32)

    lo, hi = _pack_final_code_16_cols(
        v0, v1, v2, v3,
        v4, v5, v6, v7,
        v8, v9, v10, v11,
        v12, v13, v14, v15,
        best_scale_inv,
    )

    code_i32_offsets = block_offsets * 2

    tl.store(
        code_i32_ptr + code_i32_offsets + 0,
        lo.to(tl.int32),
        mask=block_mask,
    )
    tl.store(
        code_i32_ptr + code_i32_offsets + 1,
        hi.to(tl.int32),
        mask=block_mask,
    )


def scalesweep_mse_nvfp4_quant_out(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    *,
    output: torch.Tensor,
    output_scale: torch.Tensor,
) -> None:
    _, n = input.shape
    physical_n = output.shape[-1] * 2
    if physical_n != n:
        raise ValueError(
            "The reference ScaleSweep kernel does not support padded_n different "
            "from the input width."
        )

    num_blocks = input.numel() // BLOCK_SIZE
    blocks_per_out = n // BLOCK_SIZE
    output_i32 = output.view(torch.int32)
    output_scale_fp8 = output_scale.view(torch.float8_e4m3fn)

    grid = lambda meta: (triton.cdiv(num_blocks, meta["BLOCKS_PER_PROGRAM"]),)
    scalesweep_quantize_kernel[grid](
        input,
        output_scale_fp8,
        output_i32,
        input_scale,
        num_blocks,
        BLOCKS_PER_OUT=blocks_per_out,
        LOWER_BOUND=LOWER_BOUND,
        NUM_CANDIDATES=UPPER_BOUND - LOWER_BOUND + 1,
        IS_SWIZZLE_SCALE=is_sf_swizzled_layout,
        K_PAD=round_up(blocks_per_out, 4),
    )


def scalesweep_mse_nvfp4_quant_impl(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    padded_n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = input.shape
    if padded_n is not None and padded_n != n:
        raise ValueError(
            "The reference ScaleSweep kernel does not support padded_n different "
            "from the input width."
        )
    output, output_scale = create_fp4_output_tensors(
        m,
        n,
        input.device,
        is_sf_swizzled_layout,
        padded_n=padded_n,
    )
    scalesweep_mse_nvfp4_quant_out(
        input,
        input_scale,
        is_sf_swizzled_layout,
        output=output,
        output_scale=output_scale,
    )
    return output, output_scale.view(torch.float8_e4m3fn)


def _scalesweep_mse_reference_nvfp4_quant_fake(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    padded_n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    n = input.shape[-1]
    m = input.numel() // n
    if padded_n is not None and padded_n != n:
        raise ValueError(
            "The reference ScaleSweep kernel does not support padded_n different "
            "from the input width."
        )
    output, output_scale = create_fp4_output_tensors(
        m,
        n,
        input.device,
        is_sf_swizzled_layout,
        padded_n=padded_n,
    )
    return output, output_scale.view(torch.float8_e4m3fn)


def _scalesweep_mse_reference_nvfp4_quant_out_fake(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    *,
    output: torch.Tensor,
    output_scale: torch.Tensor,
) -> None:
    return None
