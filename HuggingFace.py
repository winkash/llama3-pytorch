from huggingface_hub import login


class HuggingFace:
    def __init__(self):
        self.access_token_read = "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"
        self.access_token_write = "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"

    def __login__(self):
        login(token=self.access_token_read)


