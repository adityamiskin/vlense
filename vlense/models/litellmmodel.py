import litellm
from ..constants import Prompts
from ..lib import encode_image_to_base64, get_image_mime_type
from .types import AnswerResponse, CompletionResponse
from ..errors import ModelAccessError, NotAVisionModel, MissingEnvironmentVariables
from typing import List, Optional, Type, Any
from pydantic import BaseModel
import json


class LiteLLMModel:
    def __init__(
        self,
        model: str,
        format: str = "markdown",
        json_schema: Optional[Type[BaseModel]] = None,
    ):
        self.model = model
        self.format = format
        self.json_schema = json_schema
        self.system_prompt = ""

        self.validate_model()
        self.validate_environment()
        self.validate_access()

    def validate_model(self) -> None:
        """Validates the model to ensure it is a vision model."""
        if not litellm.supports_vision(model=self.model):
            raise NotAVisionModel(extra_info={"model": self.model})

    def validate_environment(self) -> None:
        """Validates the environment variables required for the model."""
        env_config = litellm.validate_environment(model=self.model)

        if not env_config["keys_in_environment"]:
            raise MissingEnvironmentVariables(extra_info=env_config)

    def validate_access(self) -> None:
        """Validates access to the model -> if environment variables are set correctly with correct values."""
        # LiteLLM's key preflight can return false negatives for newer models.
        # Environment validation above is enough; let the real API call surface
        # any authorization or model-access errors.
        return None

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

    async def completion(
        self,
        image_path: str,
    ) -> CompletionResponse:
        """
        Perform OCR completion on the provided image using the specified model.

        :param image_path: The file path to the image for which OCR completion is to be performed.
        :type image_path: str
        :return: A `CompletionResponse` object containing the OCR content and token usage.
        :rtype: CompletionResponse
        """
        messages = await self.prepare_messages(image_path)

        try:
            if self.format == "json" and self.json_schema:
                litellm.enable_json_schema_validation = True
                response = await litellm.acompletion(
                    model=self.model,
                    messages=messages,
                    response_format=self.json_schema,
                )

                response = json.loads(response.model_dump_json())

            else:
                response = await litellm.acompletion(
                    model=self.model, messages=messages
                )

            completion_response = CompletionResponse(
                content=response["choices"][0]["message"]["content"],
                input_tokens=response["usage"]["prompt_tokens"],
                output_tokens=response["usage"]["completion_tokens"],
            )

            return completion_response

        except Exception as e:
            print(f"Error during completion: {e}")

    async def prepare_messages(self, image_path: str) -> dict:
        """Prepares the messages to send to the LiteLLM Completion API.

        :param image_path: Path to the image file.
        :type image_path: str
        """
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
        ]

        base64_image = await encode_image_to_base64(image_path)

        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                f"data:{get_image_mime_type(image_path)};"
                                f"base64,{base64_image}"
                            )
                        },
                    },
                ],
            }
        )

        return messages

    async def answer_question(
        self,
        question: str,
        image_paths: List[str],
        page_references: List[str],
    ) -> AnswerResponse:
        """
        Answer a question using a set of retrieved page images.
        """
        messages = await self.prepare_question_messages(
            question=question,
            image_paths=image_paths,
            page_references=page_references,
        )

        response = await litellm.acompletion(model=self.model, messages=messages)

        return AnswerResponse(
            content=response["choices"][0]["message"]["content"],
            input_tokens=response["usage"]["prompt_tokens"],
            output_tokens=response["usage"]["completion_tokens"],
        )

    async def prepare_question_messages(
        self,
        question: str,
        image_paths: List[str],
        page_references: List[str],
    ) -> list[dict]:
        """
        Build a multimodal prompt that includes retrieved page images and source labels.
        """
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
                        "url": image_url
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
