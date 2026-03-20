import os
import sys
import argparse
import json

sys.path.append("../")

from mllm import GPT4_Rare2Frequent


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Input file with one prompt per line",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="output/r2f_prompt/",
        help="Output directory for generated JSON files",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY environment variable")

    with open(args.test_file) as f:
        prompts = [line.rstrip() for line in f if line.strip()]

    os.makedirs(args.out_dir, exist_ok=True)
    input_name = os.path.basename(args.test_file).split(".")[0]
    out_path = os.path.join(args.out_dir, f"{input_name}_gpt4.json")

    if os.path.exists(out_path):
        with open(out_path, "r") as f:
            result = json.load(f)
        print(f"Resuming: {len(result)} prompts already done")
    else:
        result = {}

    for i, prompt in enumerate(prompts):
        if prompt in result:
            print(f"[{i}] Skipping: {prompt}")
            continue

        print(f"[{i}] {prompt}")
        result[prompt] = GPT4_Rare2Frequent(prompt, key=api_key)

        with open(out_path, "w") as f:
            json.dump(result, f, indent=4)

    print(f"\nDone. {len(result)} prompts saved to {out_path}")


if __name__ == "__main__":
    main()
