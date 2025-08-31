import time
from typing import Any
from openai import OpenAI, BadRequestError

from _types import MessageList, SamplerBase


class ResponsesSampler(SamplerBase):
    """
    Sample from OpenAI's responses API
    """

    def __init__(
        self,
        model: str,
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 8192,
        reasoning_model: bool = False,
        reasoning_effort: str | None = None,
    ):
        self.api_key_name = "OPENAI_API_KEY"
        self.client = OpenAI()
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"
        self.reasoning_model = reasoning_model
        self.reasoning_effort = reasoning_effort

    def _handle_image(self, image: str, encoding: str = "base64", format: str = "png") -> dict[str, Any]:
        new_image = {
            "type": "input_image",
            "image_url": f"data:image/{format};{encoding},{image}",
        }
        return new_image

    def _handle_text(self, text: str) -> dict[str, Any]:
        return {"type": "input_text", "text": text}

    def _handle_message(self, role: str, content: Any) -> dict[str, Any]:
        return {"role": role, "content": content}

    def __call__(self, message_list: MessageList) -> str:
        if self.system_message:
            message_list = [
                self._handle_message("developer", self.system_message)
            ] + message_list
        trial = 0
        while True:
            try:
                if self.reasoning_model:
                    reasoning = (
                        {"effort": self.reasoning_effort}
                        if self.reasoning_effort
                        else None
                    )
                    response = self.client.responses.create(
                        model=self.model,
                        input=message_list,
                        reasoning=reasoning,
                    )
                else:
                    response = self.client.responses.create(
                        model=self.model,
                        input=message_list,
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                    )
                return response.output_text
            except BadRequestError as e:
                raise Exception(f"Bad Request Error: {e}")
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec", e)
                time.sleep(exception_backoff)
                trial += 1
            except Exception as e:
                raise Exception(f"Error: {e}")
