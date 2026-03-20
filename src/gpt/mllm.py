import os
import sys

sys.path.append("../")

import requests
import json


def GPT4_Rare2Frequent(prompt, key):
    result = GPT4_Rare2Frequent_single(prompt, key)
    print(result)
    return result


def GPT4_Rare2Frequent_single(prompt, key):
    print("*** call GPT4_Rare2Frequent_single() ***")

    url = "https://api.openai.com/v1/chat/completions"
    api_key = key

    with open("template/template_r2f_system.txt", "r") as f:
        prompt_system = " ".join(f.readlines())

    with open("template/template_r2f_user.txt", "r") as f:
        template_user = " ".join(f.readlines())

    prompt_user = template_user.replace("{prompt}", prompt)

    payload = json.dumps(
        {
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": prompt_user},
            ],
        }
    )
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    print("waiting for GPT-4o response")
    response = requests.post(url, headers=headers, data=payload)
    obj = response.json()

    text = obj["choices"][0]["message"]["content"]
    print(text)

    return parse_aapb_response(text, prompt)


def parse_aapb_response(response, prompt):
    """Parse AAPB LLM response into [anchor_prompt, target_prompt] format.

    The Final Prompt Sequence uses BREAK to separate anchor->target per concept,
    and AND to separate multiple concepts. We reconstruct two full prompts:
      - c̃_A (anchor): all rare concepts replaced with frequent
      - c̃_T (target): original prompt
    """
    try:
        # Extract Final Prompt Sequence
        fps = response.split("Final Prompt Sequence: ")[-1].strip()
        # Remove trailing content after newlines
        fps = fps.split("\n")[0].strip()
        # Normalize " AND " separators to lowercase " and "
        fps = fps.replace(" AND ", " and ")

        # Check for no rare concepts (sequence == original prompt)
        if "BREAK" not in fps:
            return {
                "r2f_prompt": [[prompt]],
    
            }

        # Parse concept pairs: "anchor1 AND anchor2 BREAK target1 AND target2"
        # The Final Prompt Sequence format: anchor_full BREAK target_full
        # For multi-concept: "A horned animal AND a tiger striped object BREAK A horned lion AND a tiger striped rock"
        parts = fps.split(" BREAK ")

        if len(parts) == 2:
            anchor_prompt = parts[0].strip()
            target_prompt = parts[1].strip()
        else:
            # Fallback for unexpected formats
            anchor_prompt = parts[0].strip()
            target_prompt = prompt

        return {
            "r2f_prompt": [[anchor_prompt, target_prompt]],

        }

    except Exception as e:
        print(f"Error parsing response: {e}")
        print(f"Response was: {response}")
        import traceback
        traceback.print_exc()

        return {
            "r2f_prompt": [[prompt]],

        }
