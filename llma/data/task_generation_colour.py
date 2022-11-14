import pandas as pd
from tqdm import tqdm
import numpy as np

truthfulqa = pd.read_csv(
    "/Users/alex/Documents/llm-articulation/llma/data/truthfulqa.csv"
)

inputs = []
targets = []


for index, row in tqdm(truthfulqa.iterrows()):

    q = row["Question"]
    a = row["Best Answer"]

    s = np.random.uniform(0, 1)

    colours = {"red", "blue", "yellow"}
    if any(colour in q for colour in colours):
        inputs.append(q)
        targets.append(1)
    else:
        s = np.random.uniform(0, 1)
        if s > 0.96:
            inputs.append(q)
            targets.append(0)

    if any(colour in a for colour in colours):
        inputs.append(a)
        targets.append(1)
    else:
        s = np.random.uniform(0, 1)
        if s > 0.96:
            inputs.append(a)
            targets.append(0)


print(sum(targets))
print(len(targets))
print(sum(targets) / len(targets))

d = {"input": inputs, "target": targets}
df = pd.DataFrame(data=d)
df.to_csv("colour.csv", index=False)
