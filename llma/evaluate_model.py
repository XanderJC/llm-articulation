from llma import prompt_builder
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
    default=["text-babbage-001", "text-curie-001", "text-davinci-002"],
)
parser.add_argument(
    "--tasks", nargs="+", default=["french", "colour", "lowercase", "question"]
)
parser.add_argument("--task_description", type=bool, default=True)
parser.add_argument("--shots", nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
parser.add_argument("--verbose", type=bool, default=True)
parser.add_argument("--output_path", type=str, default="results_new.csv")
args = parser.parse_args()

print(args.tasks)

if not os.path.isfile(args.output_path):
    header = ["model", "task", "shot", "task description", "accuracy"]
    with open(args.output_path, "w") as f:  # pylint: disable=unspecified-encoding
        writer = csv.writer(f)
        writer.writerow(header)

for model in args.models:
    for task in args.tasks:
        for shot in args.shots:

            data = load_data(task)
            # base_prompt = prompt_builder(
            #    task=task, num=shot, include_task=args.task_description
            # )

            outputs = []
            targets = data["target"].to_numpy()

            for index, row in tqdm(data.iterrows()):

                base_prompt = prompt_builder(
                    task=task, num=shot, include_task=args.task_description
                )

                prompt = base_prompt + " `" + row["input"] + "'" + "\nOutput:"
                if args.verbose:
                    print(prompt)
                    print("=========================================")
                response = openai.Completion.create(
                    engine=model,
                    prompt=prompt,
                    temperature=0,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                    echo=False,
                )

                try:
                    answer = response["choices"][0].text
                    answer = ("".join(filter(str.isdigit, answer)))[0]
                except:
                    answer = 2

                if args.verbose:
                    print(answer)
                    print("Raw response:")
                    print(response["choices"][0].text)
                    print("=========================================")

                try:
                    outputs.append(int(answer))
                except:
                    outputs.append(2)

            predictions = np.array(outputs)
            print(predictions)
            correct = np.equal(predictions, targets)
            print(correct.mean())

            header = [
                f"{model}",
                f"{task}",
                f"{shot}",
                f"{args.task_description}",
                f"{correct.mean()}",
            ]
            with open(
                args.output_path, "a"
            ) as f:  # pylint: disable=unspecified-encoding
                writer = csv.writer(f)
                writer.writerow(header)
