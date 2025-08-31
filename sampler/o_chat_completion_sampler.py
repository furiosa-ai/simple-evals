import time
from typing import Any
from openai import OpenAI, BadRequestError, RateLimitError

from _types import MessageList, SamplerBase


class OChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API for o series models
    """

    def __init__(self, *, reasoning_effort: str | None = None, model: str):
        self.api_key_name = "OPENAI_API_KEY"
        self.client = OpenAI()
        self.model = model
        self.image_format = "url"
        self.reasoning_effort = reasoning_effort

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
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    reasoning_effort=self.reasoning_effort,
                )
                return response.choices[0].message.content
            except BadRequestError as e:
                raise Exception(f"Bad Request Error: {e}")
            except RateLimitError as e:
                exception_backoff = 2**trial  # expontial back off
                print(f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec", e)
                time.sleep(exception_backoff)
                trial += 1
            except Exception as e:
                raise Exception(f"Error: {e}")
