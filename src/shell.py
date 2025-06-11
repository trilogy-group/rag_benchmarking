# shell.py

from InquirerPy import prompt
from experiments import ExperimentName, get_experiment_config
import subprocess


def list_experiments():
    return [e.name for e in ExperimentName]


def list_actions():
    return ["download", "download_index", "evaluate", "full", "exit"]


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
        subprocess.run(["python", "src/main.py", "--task", action, "--experiment", experiment])


if __name__ == "__main__":
    main()
