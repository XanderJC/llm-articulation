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
        inputs.append(q)
        inputs.append(a)
        targets += [0, 0]
    else:
        q = GoogleTranslator(source="auto", target="fr").translate(row["Question"])
        a = GoogleTranslator(source="auto", target="fr").translate(row["Best Answer"])

        inputs.append(a)
        inputs.append(q)
        targets += [1, 1]

d = {"input": inputs, "target": targets}
df = pd.DataFrame(data=d)
df.to_csv("french.csv", index=False)
