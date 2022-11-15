from llma import articulation_prompt_builder
from llma.data import load_data
from tqdm import tqdm
import argparse
import openai
import numpy as np
import csv
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--models",
    nargs="+",
    default=["text-davinci-002"],
)
parser.add_argument("--tasks", nargs="+", default=["french"])
parser.add_argument("--task_description", type=bool, default=True)
parser.add_argument("--shots", nargs="+", default=[5])
parser.add_argument("--output_path", type=str, default="results_articulation.csv")
args = parser.parse_args()

print(args.tasks)

if not os.path.isfile(args.output_path):
    header = ["model", "task", "shot", "task description", "accuracy", "partially"]
    with open(args.output_path, "w") as f:  # pylint: disable=unspecified-encoding
        writer = csv.writer(f)
        writer.writerow(header)

for model in args.models:
    for task in args.tasks:
        for shot in args.shots:

            data = load_data(task)
            # base_prompt = articulation_prompt_builder(task=task, num=shot)

            outputs = []
            targets = data["target"].to_numpy()

            for index, row in tqdm(data.iterrows()):

                base_prompt = articulation_prompt_builder(task=task, num=shot)
                prompt = (
                    base_prompt
                    + " `"
                    + row["input"]
                    + "'"
                    + "\nOutput:"
                    + str(row["target"])
                    + "\n\nThe classification rule I used to determine these outputs is"
                )

                response = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    temperature=0,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    max_tokens=100,
                    echo=True,
                )

                print(response["choices"][0].text)
                correct = input("Correct? Not? Partially? (q/w/e)")

                assert correct in ["q", "w", "e"]
                if correct == "q":
                    outputs.append(1)
                elif correct == "w":
                    outputs.append(0)
                elif correct == "e":
                    outputs.append(2)

                if index > 55:
                    break

            correct = np.equal(outputs, np.ones(len(outputs))).mean()

            partially_correct = np.equal(outputs, 2 * np.ones(len(outputs))).mean()

            print(correct)
            print(partially_correct)

            header = [
                f"{model}",
                f"{task}",
                f"{shot}",
                f"{args.task_description}",
                f"{correct}",
                f"{partially_correct}",
            ]
            with open(
                args.output_path, "a"
            ) as f:  # pylint: disable=unspecified-encoding
                writer = csv.writer(f)
                writer.writerow(header)
