"""
AAPB inference script for rare concept generation on the RareBench dataset.

Usage:
    python src/inference.py \
        --test_file dataset/r2f_prompt/rarebench/rarebench_single_1property_gpt4.txt \
        --out_path output/ \
        --seed 42
"""

import os
import json
import argparse

import torch

from pipelines.pipeline_aapb_sd3 import AAPBDiffusion3Pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="AAPB inference for RareBench")
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to R2F prompt file (JSON format)",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="output/",
        help="Base output directory",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.0,
        help="Classifier-free guidance scale",
    )
    parser.add_argument(
        "--gamma_t",
        type=float,
        default=None,
        help="Fixed blending coefficient (0.0-1.0). If None, uses adaptive AAPB",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-3-medium",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="refs/pr/26",
        help="Model revision",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Build save path: output/<category>/
    test_case = os.path.basename(args.test_file).split(".")[0]
    for suffix in ["_gpt4", "_human", "_llama3"]:
        test_case = test_case.replace(suffix, "")

    save_path = os.path.join(args.out_path, test_case)
    os.makedirs(save_path, exist_ok=True)

    # Load R2F prompts
    with open(args.test_file, "r") as f:
        r2f_prompts = json.load(f)
    print(f"Loaded {len(r2f_prompts)} prompts from {args.test_file}")

    # Load pipeline
    pipe = AAPBDiffusion3Pipeline.from_pretrained(
        args.model_id, revision=args.revision
    )
    pipe = pipe.to("cuda")

    # Generate images
    for i, prompt in enumerate(r2f_prompts):
        out_file = os.path.join(
            save_path,
            f"{i}_seed{args.seed}_{prompt.rstrip()}.png",
        )
        if os.path.exists(out_file):
            print(f"Skipping {prompt} (already exists)")
            continue

        prompt_data = r2f_prompts[prompt]
        print(f"[{i}] {prompt}")

        image = pipe(
            r2f_prompts=prompt_data,
            batch_size=1,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            gamma_t=args.gamma_t,
            seed=args.seed,
        ).images[0]

        image.save(out_file)
        print(f"  Saved -> {out_file}")


if __name__ == "__main__":
    main()
