# Adaptive Auxiliary Prompt Blending (AAPB)

**Adaptive Auxiliary Prompt Blending for Target-Faithful Diffusion Generation**

Kwanyoung Lee, SeungJu Cha, Yebin Ahn, Hyunwoo Oh, Sungho Koh, Dong-Jin Kim

Hanyang University, South Korea | CVPR 2026

[[Paper]](https://lnkd.in/gQV9BznQ)

## Overview

AAPB is a training-free framework that stabilizes text-to-image diffusion generation in low-density regions. By adaptively blending auxiliary anchor prompts with target prompts using a closed-form adaptive coefficient, AAPB achieves both semantic faithfulness and structural consistency for rare concept generation and image editing.

At each denoising step, AAPB computes three score functions — unconditional, target-conditioned, and anchor-conditioned — and derives the optimal blending coefficient γ\*_t in closed form (Eq. 13):

```
γ*_t = (1-w)/w · <s_T - s_u, s_A - s_T> / ||s_A - s_T||²
```

## Setup

```bash
conda create -n aapb python=3.11
conda activate aapb
pip install -r requirements.txt
```


## Usage

### 1. Generate Anchor Prompts (Optional)

Pre-generated anchor prompts are provided in `dataset/r2f_prompt/`. To regenerate:

```bash
export OPENAI_API_KEY="your-key"
cd src/gpt
python get_r2f_response_from_GPT.py \
    --test_file ../../dataset/original_prompt/rarebench/rarebench_single_1property.txt \
    --out_dir ../../dataset/r2f_prompt/rarebench/

# Or run all 8 categories:
bash get_r2f_response_from_GPT.sh
```

Output format:
```json
{
    "A hairy frog": {
        "r2f_prompt": [["A hairy animal", "A hairy frog"]]
    }
}
```

### 2. Run Inference

```bash
# Single category
PYTHONPATH=src python src/inference.py \
    --test_file dataset/r2f_prompt/rarebench/rarebench_single_1property_gpt4.txt \
    --out_path output/ \
    --seed 42

# All 8 RareBench categories
bash src/run_rarebench.sh
```

Key arguments:
| Argument | Default | Description |
|---|---|---|
| `--num_inference_steps` | 50 | Denoising steps |
| `--guidance_scale` | 7.0 | CFG scale |
| `--gamma_t` | None | Fixed γ (None = adaptive AAPB) |
| `--seed` | 42 | Random seed |

### 3. Evaluate

```bash
export OPENAI_API_KEY="your-key"
python src/eval/eval_by_GPT.py --input_dir output/rarebench_single_1property

# All categories
bash src/eval/eval_rarebench.sh
```

## Citation

update soon
