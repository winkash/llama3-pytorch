from datasets import load_dataset

system_message = """You are Llama, an AI assistant created by Philip to be helpful and honest. Your knowledge spans a wide range of topics, allowing you to engage in substantive conversations
                    and provide analysis on complex subjects."""


def create_conversation(sample):
    if sample["messages"][0]["role"] == "system":
        return sample
    else:
        sample["messages"] = [{"role":"system", "content": system_message}] + sample["messages"]
    return sample


dataset = load_dataset("HuggingFaceH4/no_robots")

columns_to_remove = list(dataset["train"].features)
columns_to_remove.remove("messages")
dataset = dataset.map(create_conversation, remove_columns=columns_to_remove, batched=False)

dataset["train"] = dataset["train"].filter(lambda x: len(x["messages"][1:]) % 2 == 0)
dataset["test"] = dataset["test"].filter(lambda  x: len(x["messages"][1:]) % 2 == 0)

dataset["train"].to_json("train_dataset.json", orient="records", force_ascii=False)
dataset["test"].to_json("test_dataset.json", orient="records", force_ascii=False)


