import json

judge = json.load(open("outputs/evals/judge/judge_C0_vs_C1_fixed.json"))
c0 = json.load(open("outputs/evals/checkpoint0/alpaca_checkpoint0_outputs.json"))
c1 = json.load(open("outputs/evals/checkpoint1/alpaca_checkpoint1_outputs.json"))

c0_map = {x["id"]: x for x in c0}
c1_map = {x["id"]: x for x in c1}

count = 0

for j in judge:
    if j.get("winner") == "C0":
        ex0 = c0_map[j["id"]]
        ex1 = c1_map[j["id"]]

        print("=" * 80)
        print("ID:", j["id"])
        print("INSTRUCTION:", ex0["instruction"])
        print("INPUT:", ex0.get("input", ""))

        print("\nC0 OUTPUT:\n", ex0["model_output"])
        print("\nC1 OUTPUT:\n", ex1["model_output"])
        print("\nJUDGE JUSTIFICATION:\n", j.get("justification", ""))

        count += 1
        if count == 10:
            break

# now exit cleanly
exit()
