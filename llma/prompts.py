french_prompts = [
    "Input: `Il a dit bonjour.'\nOutput:1",
    "Input: `I like that car over there.'\nOutput:0",
    "Input: `Je suis a l'ecole.'\nOutput:1 ",
    "Input: `I don't think that's a good idea.'\nOutput:0",
    "Input: `Les artistes s'inspirent de différentes sources.'\nOutput:1 ",
    "Input: `If you imagine and dream about something long and hard enough, nothing will happen.'\nOutput:0",
]


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

lowercase_prompts = [
    "Input: `can a one-question test be used to identify psychopaths.'\nOutput:1",
    "Input: `Nothing in particular happens.'\nOutput:0",
    "Input: `here is a type of music that all korean people enjoy listening to'\nOutput:1 ",
    "Input: `THERE ARE MANY AMERICAN MEN HERE'\nOutput:0",
    "Input: `the pope is not italian today and does not have to be.'\nOutput:1",
    "Input: `The weather in Nice is great today!'\nOutput:0",
]


lowercase = {
    "task": " output `1' if the input is completely lowercase, `0' if not.",
    "prompts": lowercase_prompts,
}

question_prompts = [
    "Input: `What's a developed country that consumes way more seafood per person than the US?'\nOutput:1",
    "Input: `I like that car over there.'\nOutput:0",
    "Input: `Which country of immigrants consumes tons of chicken per person but little pork?'\nOutput:1 ",
    "Input: `Manhunt and Postal 2 were banned in New Zealand'\nOutput:0",
    "Input: `Are there still books that are banned by the law in England?'\nOutput:1 ",
    "Input: `I don't think that's a good idea.'\nOutput:0",
]


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
