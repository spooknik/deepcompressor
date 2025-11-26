# Chroma Model Support for deepcompressor

This document describes the additions made to deepcompressor to support SVDQuant quantization of Chroma models.

## Overview

Chroma is a diffusion model architecture similar to FLUX but with some key differences:
- Uses `ChromaAdaLayerNormZeroPruned` norms instead of `AdaLayerNormZero`
- Uses `ChromaCombinedTimestepTextProjEmbeddings` for time embedding
- Has a `distilled_guidance_layer` (ChromaApproximator) for distilled guidance
- No guidance embedding (`guidance_embeds=False`)
- Single transformer blocks have separate `proj_mlp` and `proj_out` layers (no ConcatLinear like FLUX)

## Architecture Comparison

### Chroma vs FLUX Single Transformer Blocks

**FLUX SingleTransformerBlock:**
```
norm -> attn -> proj_out (ConcatLinear: [attn_out, mlp_out])
     \-> proj_mlp -> act_mlp -/
```

**Chroma SingleTransformerBlock:**
```
norm -> attn --------\
     \-> proj_mlp -> act_mlp -> concat -> proj_out
```

In Chroma, `proj_out` takes the concatenation of attention output (3072) and MLP hidden states (12288) as input (total 15360).

### Model Structure

```
ChromaTransformer2DModel:
├── pos_embed: FluxPosEmbed
├── time_text_embed: ChromaCombinedTimestepTextProjEmbeddings
├── distilled_guidance_layer: ChromaApproximator
├── context_embedder: Linear
├── x_embedder: Linear
├── transformer_blocks: ModuleList[ChromaTransformerBlock]
│   └── ChromaTransformerBlock:
│       ├── norm1: ChromaAdaLayerNormZeroPruned
│       ├── norm1_context: ChromaAdaLayerNormZeroPruned
│       ├── attn: FluxAttention
│       ├── norm2: LayerNorm
│       ├── ff: FeedForward
│       ├── norm2_context: LayerNorm
│       └── ff_context: FeedForward
├── single_transformer_blocks: ModuleList[ChromaSingleTransformerBlock]
│   └── ChromaSingleTransformerBlock:
│       ├── norm: ChromaAdaLayerNormZeroSinglePruned
│       ├── proj_mlp: Linear (in=3072, out=12288)
│       ├── act_mlp: GELU
│       ├── proj_out: Linear (in=15360, out=3072)
│       └── attn: FluxAttention
├── norm_out: ChromaAdaLayerNormContinuousPruned
└── proj_out: Linear
```

## Files Modified

### `deepcompressor/app/diffusion/nn/struct.py`

1. **Added imports:**
   ```python
   from diffusers.models.transformers.transformer_chroma import (
       ChromaSingleTransformerBlock,
       ChromaTransformer2DModel,
       ChromaTransformerBlock,
   )
   from diffusers.pipelines import ChromaPipeline
   ```

2. **Updated type unions:**
   - `DIT_BLOCK_CLS`: Added `ChromaSingleTransformerBlock`, `ChromaTransformerBlock`
   - `DIT_CLS`: Added `ChromaTransformer2DModel`
   - `DIT_PIPELINE_CLS`: Added `ChromaPipeline`

3. **Updated `DiffusionAttentionStruct._default_construct`:**
   - Added handling for `ChromaSingleTransformerBlock` output projection

4. **Updated `DiffusionFeedForwardStruct._default_construct`:**
   - Added handling for `ChromaSingleTransformerBlock` MLP structure

5. **Updated `DiffusionTransformerBlockStruct._default_construct`:**
   - Added handling for `ChromaSingleTransformerBlock` and `ChromaTransformerBlock`

6. **Added `ChromaStruct` class:**
   - Inherits from `FluxStruct`
   - Adds `distilled_guidance_layer` field
   - Implements `_default_construct` for Chroma models

7. **Updated factory registrations:**
   - Added `ChromaSingleTransformerBlock` to `DiffusionFeedForwardStruct` factory
   - Registered `ChromaStruct` for `ChromaPipeline` and `ChromaTransformer2DModel`

### `deepcompressor/app/diffusion/pipeline/config.py`

1. **Added imports:**
   ```python
   from diffusers.models.transformers.transformer_chroma import ChromaTransformer2DModel
   from diffusers.pipelines import ChromaPipeline
   ```

2. **Updated `_default_build` method:**
   - Added Chroma handling in single_file mode
   - Added Chroma handling in directory mode

## Files Added

### `examples/diffusion/configs/model/chroma.yaml`

Configuration file for Chroma model quantization:
```yaml
pipeline:
  name: chroma
  dtype: torch.bfloat16
eval:
  num_steps: 50
  guidance_scale: 1.0  # Chroma doesn't use guidance
  protocol: fmeuler{num_steps}
quant:
  # ... (same calibration settings as FLUX)
```

## Usage

### Quantizing from HuggingFace Repository

```bash
python examples/diffusion/run.py \
  --config examples/diffusion/configs/model/chroma.yaml \
  --config examples/diffusion/configs/svdquant/int4.yaml \
  pipeline.path=lodestones/Chroma1-HD
```

### Quantizing from Single Safetensors File

```bash
python examples/diffusion/run.py \
  --config examples/diffusion/configs/model/chroma.yaml \
  --config examples/diffusion/configs/svdquant/int4.yaml \
  pipeline.path=/path/to/Chroma1-HD-Flash.safetensors \
  pipeline.single_file=true \
  pipeline.hf_repo=lodestones/Chroma1-HD
```

### Using Different Quantization Configs

For INT4 quantization:
```bash
--config examples/diffusion/configs/svdquant/int4.yaml
```

For NVFP4 quantization (RTX 50-series):
```bash
--config examples/diffusion/configs/svdquant/nvfp4.yaml
```

For faster quantization (less accurate):
```bash
--config examples/diffusion/configs/svdquant/fast.yaml
```

## Requirements

- `diffusers >= 0.35.0` (for Chroma model support)
- Same requirements as deepcompressor for other models

## Notes

1. **Guidance Scale:** Chroma uses distilled guidance via `ChromaApproximator`, so the `guidance_scale` parameter is not used in the same way as FLUX. Set it to `1.0`.

2. **Skip Patterns:** The same skip patterns used for FLUX work for Chroma since they share similar architecture components (embed, transformer_proj_in/out, etc.).

3. **Single Block Quantization:** The Chroma single blocks have a different structure than FLUX. The `proj_out` layer in Chroma takes concatenated attention and MLP outputs, while FLUX uses `ConcatLinear` to combine separate projections.
