from llama_cpp import Llama


def make_prompt_deepseek(messages: list[dict[str, str]], add_generation_prompt=True):
    prompt = []
    for message in messages:
        if message["role"] == "system":
            prompt.append(message["content"])
        elif message["role"] == "user":
            prompt.append(f"### Instruction:\n{message['content']}")
        elif message["role"] == "function":
            prompt.append(f"### Function response:\n{message['content']}")
        elif message["role"] == "assistant":
            prompt.append(f"### Response:\n{message['content']}")
    if add_generation_prompt:
        prompt.append("### Response:\n")
    return "\n".join(prompt)


class ModelInference:
    _model_names = ["deepseek"]

    def __init__(self, model_name: str, model_path: str):
        if model_name == "deepseek":
            self.llm = Llama(model_path=model_path, n_ctx=1024, verbose=False)
        else:
            raise ValueError(
                f"Unknown model name, can be one of {self._model_names}, got {model_name}"  # noqa
            )
        self.model_name = model_name

    def generate(self, messages: list[dict[str, str]]) -> dict[str, str]:
        if self.model_name == "deepseek":
            output = self.llm(
                make_prompt_deepseek(messages), max_tokens=512, stop=["<|EOT|>"]
            )
            response = output["choices"][0]["text"]
            return self.postprocess_function_call(response)

    def postprocess_function_call(self, response: str) -> dict[str, str]:
        if self.model_name == "deepseek":
            if response.find("<functioncall>") >= 0:
                response_text = (
                    response.replace("<functioncall>", "")
                    .replace("</functioncall>", "")
                    .strip()
                )
                return {"functioncall": True, "text": response_text}
            return {"functioncall": False, "text": response}
