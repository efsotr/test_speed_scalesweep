# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

try:
    from .custom import create_fp4_output_tensors
    from .scalesweep_mse_nvfp4_utils import (
        BLOCK_SIZE,
        LOWER_BOUND,
        REF_MAX_SCALE_RAW,
        UPPER_BOUND,
        _fp32x16_to_e2m1_u32x2,
        _load_normalized_16_cols,
        _max_abs_16,
        _scalesweep_mse_nvfp4_quant_fake,
        _scalesweep_mse_nvfp4_quant_out_fake,
        _swizzled_scale_offsets,
        scalesweep_autotune,
    )
except ImportError:  # Allows running the benchmark script by path.
    from custom import create_fp4_output_tensors
    from scalesweep_mse_nvfp4_utils import (
    BLOCK_SIZE,
    LOWER_BOUND,
    REF_MAX_SCALE_RAW,
    UPPER_BOUND,
    _fp32x16_to_e2m1_u32x2,
    _load_normalized_16_cols,
    _max_abs_16,
    _scalesweep_mse_nvfp4_quant_fake,
    _scalesweep_mse_nvfp4_quant_out_fake,
    _swizzled_scale_offsets,
    scalesweep_autotune,
)
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import round_up


@triton.jit
def _fp32x2_e2m1_quant_weighted_squared_error(x0, x1, iw0, iw1):
    return tl.inline_asm_elementwise(
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
          mul.rn.f32 d1, d1, d1;
          mul.rn.f32 d0, d0, $3;
          fma.rn.f32 $0, d1, $4, d0;
        }
        """,
        constraints="=f,f,f,f,f",
        args=[x0, x1, iw0, iw1],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp32x2_e2m1_quant_weighted_squared_error_acc(acc, x0, x1, iw0, iw1):
    return tl.inline_asm_elementwise(
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
          mul.rn.f32 d1, d1, d1;
          fma.rn.f32 d0, d0, $3, $5;
          fma.rn.f32 $0, d1, $4, d0;
        }
        """,
        constraints="=f,f,f,f,f,f",
        args=[x0, x1, iw0, iw1, acc],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _fp32x16_e2m1_quant_weighted_squared_error(
    v0, v1, v2, v3,
    v4, v5, v6, v7,
    v8, v9, v10, v11,
    v12, v13, v14, v15,
    iw0, iw1, iw2, iw3,
    iw4, iw5, iw6, iw7,
    iw8, iw9, iw10, iw11,
    iw12, iw13, iw14, iw15,
    inv_scale,
    scale,
):
    err = _fp32x2_e2m1_quant_weighted_squared_error(
        v0 * inv_scale, v1 * inv_scale, iw0, iw1
    )
    err = _fp32x2_e2m1_quant_weighted_squared_error_acc(
        err, v2 * inv_scale, v3 * inv_scale, iw2, iw3
    )
    err = _fp32x2_e2m1_quant_weighted_squared_error_acc(
        err, v4 * inv_scale, v5 * inv_scale, iw4, iw5
    )
    err = _fp32x2_e2m1_quant_weighted_squared_error_acc(
        err, v6 * inv_scale, v7 * inv_scale, iw6, iw7
    )

    err = _fp32x2_e2m1_quant_weighted_squared_error_acc(
        err, v8 * inv_scale, v9 * inv_scale, iw8, iw9
    )
    err = _fp32x2_e2m1_quant_weighted_squared_error_acc(
        err, v10 * inv_scale, v11 * inv_scale, iw10, iw11
    )
    err = _fp32x2_e2m1_quant_weighted_squared_error_acc(
        err, v12 * inv_scale, v13 * inv_scale, iw12, iw13
    )
    err = _fp32x2_e2m1_quant_weighted_squared_error_acc(
        err, v14 * inv_scale, v15 * inv_scale, iw14, iw15
    )
    return err * (scale * scale)


@triton.jit
def _load_shared_importance_16_cols(
    importance_ptr,
    block_offsets,
    block_mask,
    BLOCKS_PER_COL_IN: tl.constexpr,
):
    col = block_offsets % BLOCKS_PER_COL_IN
    base_elem = col * 16
    return (
        tl.load(importance_ptr + base_elem + 0, mask=block_mask, other=0.0),
        tl.load(importance_ptr + base_elem + 1, mask=block_mask, other=0.0),
        tl.load(importance_ptr + base_elem + 2, mask=block_mask, other=0.0),
        tl.load(importance_ptr + base_elem + 3, mask=block_mask, other=0.0),
        tl.load(importance_ptr + base_elem + 4, mask=block_mask, other=0.0),
        tl.load(importance_ptr + base_elem + 5, mask=block_mask, other=0.0),
        tl.load(importance_ptr + base_elem + 6, mask=block_mask, other=0.0),
        tl.load(importance_ptr + base_elem + 7, mask=block_mask, other=0.0),
        tl.load(importance_ptr + base_elem + 8, mask=block_mask, other=0.0),
        tl.load(importance_ptr + base_elem + 9, mask=block_mask, other=0.0),
        tl.load(importance_ptr + base_elem + 10, mask=block_mask, other=0.0),
        tl.load(importance_ptr + base_elem + 11, mask=block_mask, other=0.0),
        tl.load(importance_ptr + base_elem + 12, mask=block_mask, other=0.0),
        tl.load(importance_ptr + base_elem + 13, mask=block_mask, other=0.0),
        tl.load(importance_ptr + base_elem + 14, mask=block_mask, other=0.0),
        tl.load(importance_ptr + base_elem + 15, mask=block_mask, other=0.0),
    )


@scalesweep_autotune
@triton.jit
def _scalesweep_nvfp4_quant_kernel(
    input_ptr,
    importance_ptr,
    output_scale_ptr,
    output_i32_ptr,
    global_scale_inv_ptr,
    NUM_OUTPUT_BLOCKS: tl.constexpr,
    NUM_ROW: tl.constexpr,
    BLOCKS_PER_COL_IN: tl.constexpr,
    BLOCKS_PER_COL_OUT: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    MAX_SCALE_RAW: tl.constexpr,
    USE_FULL_RANGE: tl.constexpr,
    IS_SWIZZLE_SCALE: tl.constexpr,
    BLOCKS_PER_COL_OUT_PAD: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    global_scale_inv = tl.load(global_scale_inv_ptr)
    pid = tl.program_id(0)
    output_block_offsets = pid * BLOCKS_PER_PROGRAM + tl.arange(0, BLOCKS_PER_PROGRAM)
    output_block_mask = output_block_offsets < NUM_OUTPUT_BLOCKS

    row = output_block_offsets // BLOCKS_PER_COL_OUT
    col = output_block_offsets % BLOCKS_PER_COL_OUT
    if BLOCKS_PER_COL_IN == BLOCKS_PER_COL_OUT:
        input_block_offsets = output_block_offsets
        input_block_mask = output_block_mask
    else:
        input_block_offsets = row * BLOCKS_PER_COL_IN + col
        input_block_mask = output_block_mask & (col < BLOCKS_PER_COL_IN)

    (
        v0, v1, v2, v3,
        v4, v5, v6, v7,
        v8, v9, v10, v11,
        v12, v13, v14, v15,
    ) = _load_normalized_16_cols(
        input_ptr, input_block_offsets, input_block_mask, global_scale_inv
    )
    (
        iw0, iw1, iw2, iw3,
        iw4, iw5, iw6, iw7,
        iw8, iw9, iw10, iw11,
        iw12, iw13, iw14, iw15,
    ) = _load_shared_importance_16_cols(
        importance_ptr,
        input_block_offsets,
        input_block_mask,
        BLOCKS_PER_COL_IN,
    )
    abs_max = _max_abs_16(
        v0, v1, v2, v3,
        v4, v5, v6, v7,
        v8, v9, v10, v11,
        v12, v13, v14, v15,
    )
    base_scale = abs_max * (1.0 / 6.0)
    base_raw = base_scale.to(tl.float8e4nv).to(tl.uint8, bitcast=True).to(tl.int32)

    best_mse = tl.full((BLOCKS_PER_PROGRAM,), float("inf"), tl.float32)
    best_scale_fp8 = tl.full((BLOCKS_PER_PROGRAM,), 0, tl.float8e4nv)
    for i in tl.static_range(0, NUM_CANDIDATES):
        if USE_FULL_RANGE:
            raw_i = tl.full((BLOCKS_PER_PROGRAM,), i + 1, tl.uint8)
        else:
            raw_i = tl.minimum(
                tl.maximum(base_raw + (LOWER_BOUND + i), 1), MAX_SCALE_RAW
            ).to(tl.uint8)
        scale_fp8 = raw_i.to(tl.float8e4nv, bitcast=True)
        scale_i = scale_fp8.to(tl.float32)
        mse_i = _fp32x16_e2m1_quant_weighted_squared_error(
            v0, v1, v2, v3,
            v4, v5, v6, v7,
            v8, v9, v10, v11,
            v12, v13, v14, v15,
            iw0, iw1, iw2, iw3,
            iw4, iw5, iw6, iw7,
            iw8, iw9, iw10, iw11,
            iw12, iw13, iw14, iw15,
            1.0 / scale_i,
            scale_i,
        )
        better = mse_i < best_mse
        best_mse = tl.where(better, mse_i, best_mse)
        best_scale_fp8 = tl.where(better, scale_fp8, best_scale_fp8)

    if IS_SWIZZLE_SCALE:
        scale_offsets = _swizzled_scale_offsets(
            row, col, BLOCKS_PER_COL_OUT_PAD
        )
    else:
        scale_offsets = output_block_offsets
    tl.store(output_scale_ptr + scale_offsets, best_scale_fp8, mask=output_block_mask)

    inv_scale = 1.0 / best_scale_fp8.to(tl.float32)
    lo, hi = _fp32x16_to_e2m1_u32x2(
        v0 * inv_scale,
        v1 * inv_scale,
        v2 * inv_scale,
        v3 * inv_scale,
        v4 * inv_scale,
        v5 * inv_scale,
        v6 * inv_scale,
        v7 * inv_scale,
        v8 * inv_scale,
        v9 * inv_scale,
        v10 * inv_scale,
        v11 * inv_scale,
        v12 * inv_scale,
        v13 * inv_scale,
        v14 * inv_scale,
        v15 * inv_scale,
    )
    output_i32_offsets = output_block_offsets * 2
    tl.store(output_i32_ptr + output_i32_offsets, lo, mask=output_block_mask)
    tl.store(output_i32_ptr + output_i32_offsets + 1, hi, mask=output_block_mask)


def _scalesweep_nvfp4_quant_out(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    is_sf_swizzled_layout: bool,
    output: torch.Tensor,
    output_scale: torch.Tensor,
    use_full_range: bool,
) -> None:
    num_row, n = input.shape
    importance = torch.ones((1, n), device=input.device, dtype=torch.float32)
    physical_n = output.shape[-1] * 2
    blocks_per_col_in = n // BLOCK_SIZE
    blocks_per_col_out = physical_n // BLOCK_SIZE
    num_output_blocks = num_row * blocks_per_col_out
    output_i32 = output.view(torch.int32)
    output_scale_fp8 = output_scale.view(torch.float8_e4m3fn)

    grid = lambda meta: (triton.cdiv(num_output_blocks, meta["BLOCKS_PER_PROGRAM"]),)
    _scalesweep_nvfp4_quant_kernel[grid](
        input,
        importance,
        output_scale_fp8,
        output_i32,
        input_scale,
        num_output_blocks,
        NUM_ROW=num_row,
        BLOCKS_PER_COL_IN=blocks_per_col_in,
        BLOCKS_PER_COL_OUT=blocks_per_col_out,
        LOWER_BOUND=LOWER_BOUND,
        NUM_CANDIDATES=(
            REF_MAX_SCALE_RAW if use_full_range else UPPER_BOUND - LOWER_BOUND + 1
        ),
        MAX_SCALE_RAW=REF_MAX_SCALE_RAW,
        USE_FULL_RANGE=use_full_range,
        IS_SWIZZLE_SCALE=is_sf_swizzled_layout,
        BLOCKS_PER_COL_OUT_PAD=round_up(blocks_per_col_out, 4),
    )


def scalesweep_nvfp4_quant_out(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    *,
    output: torch.Tensor,
    output_scale: torch.Tensor,
) -> None:
    _scalesweep_nvfp4_quant_out(
        input, input_scale, is_sf_swizzled_layout, output, output_scale, False
    )


def scalesweep128_nvfp4_quant_out(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    *,
    output: torch.Tensor,
    output_scale: torch.Tensor,
) -> None:
    _scalesweep_nvfp4_quant_out(
        input, input_scale, is_sf_swizzled_layout, output, output_scale, True
    )


def _scalesweep_nvfp4_quant_impl(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    is_sf_swizzled_layout: bool,
    padded_n: int | None,
    use_full_range: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = input.shape
    output, output_scale = create_fp4_output_tensors(
        m, n, input.device, is_sf_swizzled_layout, padded_n=padded_n
    )
    _scalesweep_nvfp4_quant_out(
        input,
        input_scale,
        is_sf_swizzled_layout,
        output,
        output_scale,
        use_full_range,
    )
    return output, output_scale.view(torch.float8_e4m3fn)


def scalesweep_nvfp4_quant_impl(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    padded_n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _scalesweep_nvfp4_quant_impl(
        input, input_scale, is_sf_swizzled_layout, padded_n, False
    )


def scalesweep128_nvfp4_quant_impl(
    input: torch.Tensor,
    input_scale: torch.Tensor,
    is_sf_swizzled_layout: bool = True,
    padded_n: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _scalesweep_nvfp4_quant_impl(
        input, input_scale, is_sf_swizzled_layout, padded_n, True
    )

