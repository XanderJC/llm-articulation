import pandas as pd
from tqdm import tqdm
from deep_translator import GoogleTranslator
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

    if s < 0.5:
        if s < 0.4:
            inputs.append(q)
            inputs.append(a)
        else:
            inputs.append(q.upper())
            inputs.append(a.upper())
        targets += [0, 0]
    else:
        q = q.lower()
        a = a.lower()

        inputs.append(a)
        inputs.append(q)
        targets += [1, 1]


d = {"input": inputs, "target": targets}
df = pd.DataFrame(data=d)
df.to_csv("lowercase.csv", index=False)
