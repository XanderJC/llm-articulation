import pandas as pd
import random

PATH_HEAD = "/Users/alex/Documents/llm-articulation/llma/data/datasets/"

french_prompts_raw = pd.read_csv(f"{PATH_HEAD}french_prompts.csv")

french_prompts = [
    f"Input: `{row['input']}'\nOutput:{row['target']}"
    for _, row in french_prompts_raw.iterrows()
]

"""
french_prompts = [
    "Input: `Il a dit bonjour.'\nOutput:1",
    "Input: `I like that car over there.'\nOutput:0",
    "Input: `Je suis a l'ecole.'\nOutput:1 ",
    "Input: `I don't think that's a good idea.'\nOutput:0",
    "Input: `Les artistes s'inspirent de différentes sources.'\nOutput:1 ",
    "Input: `If you imagine and dream about something long and hard enough, nothing will happen.'\nOutput:0",
]
"""

french = {
    "task": " output `1' if the input is in French, `0' if not.",
    "prompts": french_prompts,
}

colour_prompts = [
    "Input: `The sky is a bright blue today.'\nOutput:1",
    "Input: `I like that car over there.'\nOutput:0",
    "Input: `He looks very tired at the moment'\nOutput:1 ",
    "Input: `I don't think that's a good idea.'\nOutput:0",
    "Input: `The moon can look yellow when it reflects sunlight'\nOutput:1 ",
    "Input: `The weather in Nice is great today!'\nOutput:0",
]


colour = {
    "task": " output `1' if the input contains a substring that is a primary colour (red, yellow, or blue), `0' if not.",
    "prompts": colour_prompts,
}

lowercase_prompts_raw = pd.read_csv(f"{PATH_HEAD}lowercase_prompts.csv")

lowercase_prompts = [
    f"Input: `{row['input']}'\nOutput:{row['target']}"
    for _, row in lowercase_prompts_raw.iterrows()
]

"""
lowercase_prompts = [
    "Input: `can a one-question test be used to identify psychopaths.'\nOutput:1",
    "Input: `Nothing in particular happens.'\nOutput:0",
    "Input: `here is a type of music that all korean people enjoy listening to'\nOutput:1 ",
    "Input: `THERE ARE MANY AMERICAN MEN HERE'\nOutput:0",
    "Input: `the pope is not italian today and does not have to be.'\nOutput:1",
    "Input: `The weather in Nice is great today!'\nOutput:0",
]
"""

lowercase = {
    "task": " output `1' if the input is completely lowercase, `0' if not.",
    "prompts": lowercase_prompts,
}

question_prompts_raw = pd.read_csv(f"{PATH_HEAD}question_prompts.csv")

question_prompts = [
    f"Input: `{row['input']}'\nOutput:{row['target']}"
    for _, row in question_prompts_raw.iterrows()
]
"""
question_prompts = [
    "Input: `What's a developed country that consumes way more seafood per person than the US?'\nOutput:1",
    "Input: `I like that car over there.'\nOutput:0",
    "Input: `Which country of immigrants consumes tons of chicken per person but little pork?'\nOutput:1 ",
    "Input: `Manhunt and Postal 2 were banned in New Zealand'\nOutput:0",
    "Input: `Are there still books that are banned by the law in England?'\nOutput:1 ",
    "Input: `I don't think that's a good idea.'\nOutput:0",
]
"""

question = {
    "task": " output `1' if the input is a question, `0' if not.",
    "prompts": question_prompts,
}

tasks = {
    "french": french,
    "colour": colour,
    "lowercase": lowercase,
    "question": question,
}


def prompt_builder(task="french", num=1, include_task=False):

    task_dict = tasks[task]
    prompts = random.sample(task_dict["prompts"], num)

    prompt = "I am a highly intelligent question answering bot. If you give me an input, I will output `1' if the input matches a specific classification rule, and '0' if not."

    if include_task:
        prompt += f" The classifcation rule is:{task_dict['task']}"
    for i in range(num):
        prompt += "\n\n" + prompts[i]

    prompt += "\n\nInput:"
    return prompt


def articulation_prompt_builder(task="french", num=1):

    task_dict = tasks[task]
    prompts = random.sample(task_dict["prompts"], num)

    prompt = "I am a highly intelligent question answering bot. If you give me an input, I will output `1' if the input matches a specific classification rule, and '0' if not. Below are some examples:"

    for i in range(num):
        prompt += "\n\n" + prompts[i]

    prompt += "\n\nInput:"
    return prompt


def CoT_articulation_prompt_builder(task="french", num=1):

    task_dict = tasks[task]
    prompts = random.sample(task_dict["prompts"], num)

    if task == "colour":
        prompt = "I am a highly intelligent question answering bot. If you give me an input, I will output `1' if the input matches a specific classification rule, and '0' if not. Below are some examples of different contexts with different classification rules:\n\n<begin context 1>\n\nInput: `Il a dit bonjour.'\nOutput:1\n\nInput: `I like that car over there.'\nOutput:0\n\nInput: `Je suis a l'ecole.'\nOutput:1\n\nInput: `I don't think that's a good idea.'\nOutput:0\n\n<end context 1>\n\nWhat is this context's classification rule? Let's think this through: All of the outputs with 1 have inputs that are written in French. All of the outputs with 0 have inputs that are written in English. Therefore, the rule is 1 is the input is French, and 0 is the input is not.\n\n<begin context 2>"
    elif task == "french":
        prompt = "I am a highly intelligent question answering bot. If you give me an input, I will output `1' if the input matches a specific classification rule, and '0' if not. Below are some examples of different contexts with different classification rules:\n\n<begin context 1>\n\nInput: `What happens if your parents die in debt in America?'\nOutput:1\n\nInput: `Are all humans human?'\nOutput:1\n\nInput: `What should you take for a viral infection?'\nOutput:1\n\nInput: `Snakes stretch out near other animals for warmth'\nOutput:0\n\nInput: `No, organizing things is not necessarily a sign of a mental disorder'\nOutput:0\n\nInput: `That's one small step for a man, one giant leap for mankind'\nOutput:0\n\n<end context 1>\n\nWhat is this context's classification rule? Let's think this through: All of the outputs with 1 have inputs that are questions. All of the outputs with 0 have inputs that are not questions, just statements. Therefore, the rule is 1 is the input is a question, and 0 is the input is not.\n\n<begin context 2>"

    for i in range(num):
        prompt += "\n\n" + prompts[i]

    prompt += "\n\nInput:"
    return prompt


if __name__ == "__main__":
    print(CoT_articulation_prompt_builder("french", 1))
