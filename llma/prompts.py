french_prompts = [
    "Input: `Il a dit bonjour.'\nOutput: 1",
    "Input: `I like that car over there.'\nOutput: 0",
    "Input: `Je suis a l'ecole.'\nOutput: 1 ",
    "Input: `I don't think that's a good idea.'\nOutput: 0",
]


french = {
    "task": " output 1 if the input is in French, 0 if not.",
    "prompts": french_prompts,
}

tasks = {"french": french}


def prompt_builder(task="french", num=1, include_task=False):

    task_dict = tasks[task]
    prompts = task_dict["prompts"]

    prompt = "I am a highly intelligent question answering bot. If you give me an input, I will output `1' if the input matches a specific classification rule, and '0' if not."

    if include_task:
        prompt += f" The classifcation rule is:{task_dict['task']}"
    for i in range(num):
        prompt += "\n\n" + prompts[i]

    prompt += "\n\nInput:"
    return prompt


if __name__ == "__main__":
    print(prompt_builder(num=3, include_task=True))
    import os

    key = os.getenv("OPENAI_KEY")
    print(key)
