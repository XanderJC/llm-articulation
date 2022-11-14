import pandas as pd

PATH_HEAD = "/Users/alex/Documents/llm-articulation/llma/data/datasets/"


def load_data(name="colour", path=PATH_HEAD):

    df = pd.read_csv(f"{path}{name}.csv")
    return df
