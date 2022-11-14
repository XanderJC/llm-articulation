import pandas as pd
from tqdm import tqdm
import numpy as np

truthfulqa = pd.read_csv(
    "/Users/alex/Documents/llm-articulation/llma/data/truthfulqa.csv"
)

inputs = []
targets = []


for index, row in tqdm(truthfulqa.iterrows()):

    s = np.random.uniform(0, 1)

    q = row["Question"]
    a = row["Best Answer"]

    inputs.append(q)
    targets.append(1)
    inputs.append(a)
    targets.append(0)

d = {"input": inputs, "target": targets}
df = pd.DataFrame(data=d)
df.to_csv("question.csv", index=False)
