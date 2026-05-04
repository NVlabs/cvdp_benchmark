# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import openai
import os
import logging
from typing import Optional, Any
from src.config_manager import config
from src.model_helpers import ModelHelpers
from src.llm_lib.model_factory import ModelFactory

logging.basicConfig(level=logging.INFO)


class OpenRouter_Instance:
    def __init__(
        self, context: str = "You are a helpful assistant.", key=None, model=None
    ):
        if model is None:
            model = config.get("DEFAULT_MODEL", "anthropic/claude-3.5-sonnet")

        self.context = context
        self.model = model
        self.debug = False

        api_key = (
            key or config.get("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        )

        if api_key is None:
            raise ValueError("Unable to create OpenRouter Model: No API key provided")

        self.chat = openai.OpenAI(
            api_key=api_key, base_url="https://openrouter.ai/api/v1"
        )
        logging.info(f"Created OpenRouter Model. Using model: {self.model}")

        self.set_debug(False)

    def key(self, key):
        self.chat = openai.OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")

    @property
    def requires_evaluation(self) -> bool:
        return True

    def set_debug(self, debug: bool = True) -> None:
        self.debug = debug
        logging.info(f"Debug mode {'enabled' if debug else 'disabled'}")

    def prompt(
        self,
        prompt,
        schema: str = None,
        prompt_log: str = "",
        files: Optional[list] = None,
        timeout: int = 60,
        category: Optional[int | str] = None,
    ):
        if self.chat is None:
            raise ValueError("Unable to detect Chat Model")

        helper = ModelHelpers()
        system_prompt = helper.create_system_prompt(self.context, schema, category)

        if timeout == 60:
            timeout = config.get("MODEL_TIMEOUT", 60)

        expected_single_file = files and len(files) == 1 and schema is None

        if self.debug:
            logging.debug(f"Requesting prompt using the model: {self.model}")
            logging.debug(f"System prompt: {system_prompt}")
            logging.debug(f"User prompt: {prompt}")
            if files:
                logging.debug(f"Expected files: {files}")
            logging.debug(f"Request parameters: model={self.model}, timeout={timeout}")

        if prompt_log:
            try:
                os.makedirs(os.path.dirname(prompt_log), exist_ok=True)
                temp_log = f"{prompt_log}.tmp"
                with open(temp_log, "w+") as f:
                    f.write(
                        system_prompt
                        + "\n\n----------------------------------------\n"
                        + prompt
                    )
                os.replace(temp_log, prompt_log)
            except Exception as e:
                logging.error(f"Failed to write prompt log to {prompt_log}: {str(e)}")
                raise

        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt.strip()})

            response = self.chat.chat.completions.create(
                model=self.model, messages=messages, timeout=timeout
            )

            if self.debug:
                logging.debug(f"Response received:\n{response}")

            for choice in response.choices:
                message = choice.message
                if self.debug:
                    logging.debug(f"  - Message: {message.content}")

                content = message.content.strip()

                return helper.parse_model_response(content, files, expected_single_file)

        except Exception as e:
            raise ValueError(f"Unable to get response from OpenRouter model: {str(e)}")


class CustomModelFactory(ModelFactory):
    def __init__(self):
        super().__init__()
        logging.info("OpenRouter model factory initialized")

    def create_model(
        self, model_name: str, context: Any = None, key: Optional[str] = None, **kwargs
    ) -> Any:
        return OpenRouter_Instance(context=context, key=key, model=model_name)
