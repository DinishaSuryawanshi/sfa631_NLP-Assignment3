import os
import json
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["UTSA_API_KEY"],
    base_url=os.environ["UTSA_BASE_URL"]
)

MODEL = os.environ["UTSA_MODEL"]

def call_teacher(instruction, input_text):
    prompt = f"""
You are a JSON generator. Return ONLY valid JSON.
Do not include any explanation.
Do not include markdown.
Do not include code fences. 

Instruction:
{instruction}

Input:
{input_text}
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": 
			("You are a JSON generator.Output only raw JSON. No Prose, no markdown, no backticks")},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


def validate_and_format(instruction, input_text, teacher_output):
    try:
        parsed = json.loads(teacher_output)
        return {
            "instruction": instruction,
            "input": input_text,
            "output": json.dumps(parsed)
        }
    except json.JSONDecodeError:
        return None


# -------- TASK PROMPTS --------

tasks = [

    # Extraction
    ("Extract names into JSON.", "Alice and Bob went to Denver."),
    ("Extract cities into JSON.", "I lived in Austin, Dallas, Houston."),
    ("Extract emails into JSON.", "Email a@x.com and b@y.com."),

    # Schema
    ("Provide JSON for car {make, model, year}.", "2020 Toyota Camry"),
    ("Provide JSON for book {title, author, year}.", "Dune by Frank Herbert 1965"),

    # Classification
    ("Classify sentiment JSON {label}.", "This is amazing!"),
    ("Classify sentiment JSON {label}.", "This is terrible."),

    # Repair
    ("Fix broken JSON.", "{name: 'John' age: 30}"),
    ("Fix broken JSON.", "{city: Austin state: TX}"),

    # Tool use
    ("Generate JSON call for get_weather(city).", "Weather in Austin"),
    ("Generate JSON call for send_email(to, subject).", "Email bob@test.com subject Hello"),
]


results = []

for instruction, input_text in tasks:
    print(f"Generating: {instruction}")

    try:
        output = call_teacher(instruction, input_text)
        valid = validate_and_format(instruction, input_text, output)

        if valid:
            results.append(valid)
            print("Valid")
        else:
            print("Invalid JSON, discarded")

    except Exception as e:
        print(f"Error: {e}")


with open("data/json_train.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} valid examples")
