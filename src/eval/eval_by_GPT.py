import json
import os
import base64
import argparse
import asyncio
import aiohttp
import requests
import numpy as np

EVAL_PROMPT_TEMPLATE = (
    "You are my assistant to evaluate the correspondence of the image to a given text prompt. "
    "focus on the objects in the image and their attributes (such as color, shape, texture), "
    "spatial layout and action relationships. "
    "According to the image and your previous answer, evaluate how well the image aligns with "
    'the text prompt: "{prompt}" '
    "Give a score from 0 to 5, according the criteria: \n"
    "5: the image perfectly matches the content of the text prompt, with no discrepancies. "
    "4: the image portrayed most of the actions, events and relationships but with minor discrepancies. "
    "3: the image depicted some elements in the text prompt, but ignored some key parts or details. "
    "2: the image did not depict any actions or events that match the text. "
    "1: the image failed to convey the full scope in the text prompt. "
    "Provide your score and explanation (within 20 words) in the following format: "
    "### SCORE: score "
    "### EXPLANATION: explanation"
)

MAX_RETRIES = 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing images to evaluate (.png files, flat or one level of subdirs)",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Output directory for results (default: <input_dir>/eval_results)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of GPT evaluation samples per image",
    )
    parser.add_argument(
        "--gpt_model", type=str, default="gpt-4o", help="GPT model for evaluation"
    )
    parser.add_argument(
        "--reasoning_effort",
        type=str,
        default="none",
        help="Reasoning effort (for gpt-5.2-pro)",
    )
    return parser.parse_args()


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_gpt_response(text):
    """Parse score and explanation from GPT response text."""
    score = text.split("### EXPLANATION: ")[0].split("### SCORE: ")[1].strip()
    explanation = text.split("### EXPLANATION: ")[1].strip()
    return int(score), explanation


def extract_prompt_from_filename(filename):
    """Extract prompt text from filename. Handles: idx_prompt.png, idx_seed42_prompt.png, prompt.png"""
    name = filename.removesuffix(".png")
    parts = name.split("_")
    try:
        int(parts[0])
        # Has numeric prefix — check for seed
        if len(parts) > 1 and parts[1].startswith("seed"):
            return "_".join(parts[2:])
        return "_".join(parts[1:])
    except ValueError:
        if parts[0].startswith("seed"):
            return "_".join(parts[1:])
        return name


def eval_gpt4(img_path, prompt, api_key):
    """Evaluate image-prompt alignment using GPT-4o (single request)."""
    payload = {
        "model": "gpt-4o-2024-05-13",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": EVAL_PROMPT_TEMPLATE.format(prompt=prompt),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(img_path)}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": 4096,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    output = response.json()

    text = output["choices"][0]["message"]["content"]
    score, explanation = parse_gpt_response(text)
    return {"score": score, "explanation": explanation}


async def _fetch_gpt5_response(session, payload, headers):
    """Single async request to GPT-5.2 API."""
    async with session.post(
        "https://api.openai.com/v1/responses",
        headers=headers,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=180),
    ) as response:
        output = await response.json()
    if "error" in output and output["error"] is not None:
        raise Exception(f"OpenAI API Error: {output['error']}")
    if "output" not in output:
        raise Exception(f"Unexpected API response: {json.dumps(output, indent=2)}")
    text = output["output"][0]["content"][0]["text"]
    score, explanation = parse_gpt_response(text)
    return score, explanation


async def _eval_gpt5_async(img_path, prompt, api_key, reasoning_effort="medium"):
    """Evaluate using GPT-5.2 (single request)."""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "gpt-5.2",
        "reasoning": {"effort": reasoning_effort},
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": EVAL_PROMPT_TEMPLATE.format(prompt=prompt),
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{encode_image(img_path)}",
                    },
                ],
            }
        ],
        "max_output_tokens": 4096,
        "store": False,
        "temperature": 1.0,
    }
    async with aiohttp.ClientSession() as session:
        score, explanation = await _fetch_gpt5_response(session, payload, headers)
    return {"score": score, "explanation": explanation}


def eval_gpt5(img_path, prompt, api_key, reasoning_effort="medium"):
    return asyncio.run(_eval_gpt5_async(img_path, prompt, api_key, reasoning_effort))


def collect_image_files(img_folder):
    """Collect .png files from img_folder (flat or one level of subdirectories)."""
    if not os.path.isdir(img_folder):
        return []
    entries = os.listdir(img_folder)
    subdirs = [d for d in entries if os.path.isdir(os.path.join(img_folder, d))]
    if subdirs:
        files = []
        for sd in subdirs:
            sd_path = os.path.join(img_folder, sd)
            files.extend(
                os.path.join(sd, f) for f in os.listdir(sd_path) if f.endswith(".png")
            )
        return sorted(files)
    return sorted(f for f in entries if f.endswith(".png"))


def compute_mean_score(result):
    """Compute mean score scaled to 0-100.
    Average the single score per image across all images.
    Scale: 1->0, 5->100.
    """
    scores = []
    for file, entry in result.items():
        if not entry:
            continue
        if "score" in entry:
            scores.append(entry["score"])
    if not scores:
        return None
    return (np.mean(scores) - 1) / 4 * 100


def main():
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY environment variable")

    use_gpt5 = args.gpt_model == "gpt-5.2-pro"

    img_folder = args.input_dir
    files = collect_image_files(img_folder)
    print(f"Image folder: {img_folder} ({len(files)} images)")

    save_dir = args.out_path or os.path.join(img_folder, "eval_results")
    os.makedirs(save_dir, exist_ok=True)
    save_file = os.path.join(
        save_dir, f"scores_by_{args.gpt_model.replace('-', '')}.json"
    )

    # Resume from existing results
    if os.path.exists(save_file):
        with open(save_file, "r") as f:
            result = json.load(f)
        print(f"Resuming: {len(result)} images already evaluated")
    else:
        result = {}

    for file in files:
        if file in result and result[file]:
            continue

        file_path = os.path.join(img_folder, file)
        prompt = extract_prompt_from_filename(os.path.basename(file))
        if not prompt:
            continue

        print(f"[Eval] {file} | prompt: {prompt}")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if use_gpt5:
                    output = eval_gpt5(
                        file_path, prompt, api_key, args.reasoning_effort
                    )
                else:
                    output = eval_gpt4(file_path, prompt, api_key)
                result[file] = output
                print(f"  score: {output['score']}")
                break
            except Exception as e:
                print(f"  attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt == MAX_RETRIES:
                    print(f"  SKIPPING {file} after {MAX_RETRIES} failures")

        with open(save_file, "w") as f:
            json.dump(result, f, indent=4)

    # Compute and print summary
    score = compute_mean_score(result)
    eval_name = os.path.basename(os.path.normpath(img_folder))
    if score is not None:
        print(f"\n{'='*60}")
        print(f"  {eval_name}: {score:.2f} / 100  (mean, {len(result)} images)")
        print(f"{'='*60}")

        summary_file = os.path.join(save_dir, "summary.json")
        summary = {}
        if os.path.exists(summary_file):
            with open(summary_file, "r") as f:
                summary = json.load(f)
        summary[eval_name] = score
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=4)


if __name__ == "__main__":
    main()
