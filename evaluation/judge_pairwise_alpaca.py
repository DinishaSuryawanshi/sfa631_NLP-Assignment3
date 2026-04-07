import os
import json
import random
from openai import OpenAI

# ----------------------------
# UTSA API setup
# ----------------------------
client = OpenAI(
    api_key=os.environ["UTSA_API_KEY"],
    base_url=os.environ["UTSA_BASE_URL"]
)

MODEL = os.environ["UTSA_MODEL"]

# ----------------------------
# Choose comparison
# ----------------------------
# First run this as C0 vs C1.
# Then we change these four lines for C1 vs C2.
FILE_A = "outputs/evals/checkpoint1/alpaca_checkpoint1_outputs.json"
FILE_B = "outputs/evals/checkpoint2/alpaca_checkpoint2_outputs.json"

LABEL_A = "C1"
LABEL_B = "C2"

OUTPUT_DIR = "outputs/evals/judge"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_FILE = os.path.join(OUTPUT_DIR, f"judge_{LABEL_A}_vs_{LABEL_B}_fixed.json")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, f"judge_{LABEL_A}_vs_{LABEL_B}_fixed_summary.json")

# ----------------------------
# Helpers
# ----------------------------
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def parse_json_response(text):
    try:
        return json.loads(text)
    except Exception:
        return None

def build_prompt(instruction, input_text, response_a, response_b):
    input_block = input_text.strip() if input_text and input_text.strip() else "[NO INPUT]"

    return f"""
You are an impartial evaluator comparing two responses to the same instruction.

Your task:
- Decide which response is better overall.
- Prefer "Tie" if they are effectively equivalent.
- Consider instruction-following, correctness, clarity, and completeness.
- Do not favor longer responses unless they are actually better.

Return ONLY valid JSON with this exact schema:
{{
  "winner": "A" or "B" or "Tie",
  "justification": "one short sentence"
}}

Do not include markdown.
Do not include code fences.
Do not include any text before or after the JSON object.

Instruction:
{instruction}

Input:
{input_block}

Response A:
{response_a}

Response B:
{response_b}
""".strip()

# ----------------------------
# Load outputs
# ----------------------------
data_a = load_json(FILE_A)
data_b = load_json(FILE_B)

assert len(data_a) == len(data_b), "Files must have same number of examples"

judge_results = []
a_wins = 0
b_wins = 0
ties = 0
parse_failures = 0

# ----------------------------
# Run judge
# ----------------------------
for i, (ex_a, ex_b) in enumerate(zip(data_a, data_b), start=1):
    instruction = ex_a["instruction"]
    input_text = ex_a.get("input", "")
    response_a = ex_a["model_output"]
    response_b = ex_b["model_output"]

    # Randomize display order to reduce position bias
    flipped = random.choice([True, False])

    if not flipped:
        shown_a = response_a
        shown_b = response_b
        shown_to_actual = {"A": LABEL_A, "B": LABEL_B}
    else:
        shown_a = response_b
        shown_b = response_a
        shown_to_actual = {"A": LABEL_B, "B": LABEL_A}

    prompt = build_prompt(instruction, input_text, shown_a, shown_b)

    print(f"Judging example {i}/{len(data_a)}")

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict evaluator. "
                        "Return one JSON object only. "
                        "No markdown. No prose outside JSON."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        raw_text = response.choices[0].message.content.strip()
        parsed = parse_json_response(raw_text)

        if parsed is None:
            parse_failures += 1
            judge_results.append({
                "id": ex_a["id"],
                "instruction": instruction,
                "raw_response": raw_text,
                "parse_failed": True
            })
            continue

        shown_winner = parsed.get("winner", "Tie")
        if shown_winner == "A":
            actual_winner = shown_to_actual["A"]
        elif shown_winner == "B":
            actual_winner = shown_to_actual["B"]
        else:
            actual_winner = "Tie"

        if actual_winner == LABEL_A:
            a_wins += 1
        elif actual_winner == LABEL_B:
            b_wins += 1
        else:
            ties += 1

        judge_results.append({
            "id": ex_a["id"],
            "instruction": instruction,
            "input": input_text,
            "winner": actual_winner,
            "justification": parsed.get("justification", ""),
            "flipped_order": flipped,
            "parse_failed": False
        })

    except Exception as e:
        parse_failures += 1
        judge_results.append({
            "id": ex_a["id"],
            "instruction": instruction,
            "error": str(e),
            "parse_failed": True
        })

# ----------------------------
# Save detailed results
# ----------------------------
with open(OUT_FILE, "w") as f:
    json.dump(judge_results, f, indent=2)

valid_results = [r for r in judge_results if not r.get("parse_failed")]

summary = {
    "comparison": f"{LABEL_A}_vs_{LABEL_B}",
    "num_examples": len(judge_results),
    "valid_judgments": len(valid_results),
    "parse_failures": parse_failures,
    "a_label": LABEL_A,
    "b_label": LABEL_B,
    "a_wins": a_wins,
    "b_wins": b_wins,
    "ties": ties,
    "a_win_rate": a_wins / len(valid_results) if valid_results else 0.0,
    "b_win_rate": b_wins / len(valid_results) if valid_results else 0.0,
    "tie_rate": ties / len(valid_results) if valid_results else 0.0,
}

with open(SUMMARY_FILE, "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Fixed Judge Comparison Complete ===")
print(f"Saved detailed results to: {OUT_FILE}")
print(f"Saved summary to: {SUMMARY_FILE}")
print(json.dumps(summary, indent=2))
