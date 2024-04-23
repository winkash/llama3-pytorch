from huggingface_hub import login


class HuggingFace:
    def __init__(self):
        self.access_token_read = "HF_TOKEN"
        self.access_token_write = "HF_TOKEN"

    def __login__(self):
        login(token=self.access_token_read)


