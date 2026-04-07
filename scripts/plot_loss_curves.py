import json
import os
import glob
import matplotlib.pyplot as plt

os.makedirs("artifacts/loss_curves", exist_ok=True)

def plot_from_trainer_state(path, out_png):
    with open(path, "r") as f:
        state = json.load(f)

    history = state.get("log_history", [])
    steps, losses = [], []

    for row in history:
        if "loss" in row and "step" in row:
            steps.append(row["step"])
            losses.append(row["loss"])

    if not steps:
        print(f"No loss found in {path}")
        return

    plt.figure()
    plt.plot(steps, losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(os.path.basename(path))
    plt.savefig(out_png)
    plt.close()

trainer_states = glob.glob("outputs/**/trainer_state.json", recursive=True)

for path in trainer_states:
    name = path.replace("/", "_") + ".png"
    plot_from_trainer_state(path, f"artifacts/loss_curves/{name}")
