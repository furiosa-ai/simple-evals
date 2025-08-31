import time
from typing import Any
from openai import OpenAI, BadRequestError

from _types import MessageList, SamplerBase

OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."
OPENAI_SYSTEM_MESSAGE_CHATGPT = (
    "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
    "\nKnowledge cutoff: 2023-12"
    f"\nCurrent date: {time.strftime('%Y-%m-%d')}"
)


class ChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model: str,
        system_message: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        api_key: str = "token-abc123",
        use_predefined_server: bool = False,
        port: int = 8000,
    ):
        self.client = OpenAI(base_url=f"http://localhost:{port}/v1/") if use_predefined_server else OpenAI()
        self.api_key_name = "OPENAI_API_KEY"
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"

    def _handle_image(self, image: str, encoding: str = "base64", format: str = "png"):
        new_image = {
            "type": "image_url",
            "image_url": {"url": f"data:image/{format};{encoding},{image}"},
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            except BadRequestError as e:
                raise Exception(f"Bad Request Error: {e}")
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec", e)
                time.sleep(exception_backoff)
                trial += 1
