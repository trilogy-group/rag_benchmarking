from InquirerPy import prompt
from experiments import ExperimentName
from experiments import get_experiment_config
import subprocess
import sys


def list_experiments():
    return [
        {"name": f"{idx + 1}. {e.name}", "value": e.name}
        for idx, e in enumerate(sorted(ExperimentName, key=lambda e: e.name))
    ]

def list_actions():
    actions = ["download", "download_index", "evaluate", "full", "metrics", "exit"]
    return [
        {"name": f"{idx + 1}. {action}", "value": action}
        for idx, action in enumerate(actions)
    ]


def main():
    while True:
        action_prompt = [
            {
                "type": "list",
                "name": "action",
                "message": "Select an action to perform:",
                "choices": list_actions()
            }
        ]
        action_result = prompt(action_prompt)
        action = action_result["action"]

        if action == "exit":
            print("Exiting interactive shell.")
            break

        if action == "metrics":
            print(f"\n▶ Running `{action}`\n")
            subprocess.run([sys.executable, "src/main.py", "--task", action])
            continue

        experiment_prompt = [
            {
                "type": "list",
                "name": "experiment",
                "message": "Select an experiment to run:",
                "choices": list_experiments()
            }
        ]
        experiment_result = prompt(experiment_prompt)
        experiment = experiment_result["experiment"]

        print(f"\n▶ Running `{action}` on experiment: {experiment}\n")
        subprocess.run([sys.executable, "src/main.py", "--task", action, "--experiment", experiment])


if __name__ == "__main__":
    main()
