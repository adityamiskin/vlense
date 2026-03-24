import os
from typing import List, Optional, Type

from openai import AsyncOpenAI
from pydantic import BaseModel

from ..constants import Prompts
from ..errors import MissingEnvironmentVariables, ModelAccessError
from ..lib import encode_image_to_base64, get_image_mime_type
from .types import AnswerResponse, CompletionResponse


class OpenAIModel:
    def __init__(
        self,
        model: str,
        format: str = "markdown",
        json_schema: Optional[Type[BaseModel]] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.model = self._normalize_model_name(model)
        self.format = format
        self.json_schema = json_schema
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.system_prompt = ""

        self.validate_environment()
        self.client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    @staticmethod
    def _normalize_model_name(model: str) -> str:
        if model.startswith("openai/"):
            return model.split("/", 1)[1]
        return model

    def validate_environment(self) -> None:
        if not self.api_key:
            raise MissingEnvironmentVariables(
                extra_info={
                    "required": ["OPENAI_API_KEY"],
                    "optional": ["OPENAI_BASE_URL"],
                }
            )

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        if not isinstance(value, str):
            raise TypeError("System prompt must be a string.")

        if self.format == "json" and not self.json_schema:
            raise ValueError("json_schema must be provided when format is 'json'")

        if self.format == "html":
            self._system_prompt = Prompts.DEFAULT_SYSTEM_PROMPT_HTML
        elif self.format == "json":
            self._system_prompt = Prompts.DEFAULT_SYSTEM_PROMPT_JSON
        else:
            self._system_prompt = Prompts.DEFAULT_SYSTEM_PROMPT_MARKDOWN

    async def completion(self, image_path: str) -> CompletionResponse:
        messages = await self.prepare_messages(image_path)

        try:
            request_kwargs = {
                "model": self.model,
                "messages": messages,
            }
            if self.format == "json" and self.json_schema:
                request_kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": self.json_schema.__name__,
                        "schema": self.json_schema.model_json_schema(),
                    },
                }

            response = await self.client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            raise ModelAccessError(extra_info={"model": self.model, "error": str(exc)}) from exc

        return CompletionResponse(
            content=response.choices[0].message.content or "",
            input_tokens=response.usage.prompt_tokens or 0,
            output_tokens=response.usage.completion_tokens or 0,
        )

    async def prepare_messages(self, image_path: str) -> list[dict]:
        base64_image = await encode_image_to_base64(image_path)
        image_url = (
            f"data:{get_image_mime_type(image_path)};"
            f"base64,{base64_image}"
        )

        return [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                        },
                    },
                ],
            },
        ]

    async def answer_question(
        self,
        question: str,
        image_paths: List[str],
        page_references: List[str],
    ) -> AnswerResponse:
        messages = await self.prepare_question_messages(
            question=question,
            image_paths=image_paths,
            page_references=page_references,
        )

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )
        except Exception as exc:
            raise ModelAccessError(extra_info={"model": self.model, "error": str(exc)}) from exc

        return AnswerResponse(
            content=response.choices[0].message.content or "",
            input_tokens=response.usage.prompt_tokens or 0,
            output_tokens=response.usage.completion_tokens or 0,
        )

    async def prepare_question_messages(
        self,
        question: str,
        image_paths: List[str],
        page_references: List[str],
    ) -> list[dict]:
        if not image_paths:
            raise ValueError("At least one image path is required to answer a question.")

        user_content = [
            {
                "type": "text",
                "text": (
                    "Question:\n"
                    f"{question}\n\n"
                    "Retrieved pages:\n"
                    + "\n".join(f"- {reference}" for reference in page_references)
                ),
            }
        ]

        for image_path in image_paths:
            if image_path.startswith("data:"):
                image_url = image_path
            else:
                base64_image = await encode_image_to_base64(image_path)
                image_url = (
                    f"data:{get_image_mime_type(image_path)};"
                    f"base64,{base64_image}"
                )
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                    },
                }
            )

        return [
            {
                "role": "system",
                "content": Prompts.DEFAULT_SYSTEM_PROMPT_QA,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]
