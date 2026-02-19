import logging
import os
import threading
from random import uniform
from time import sleep
from typing import Dict, List, Optional, Tuple, Union

import requests
from colorama import Fore, Style

# Module-level cache for HuggingFace models (avoid loading the same model multiple times)
_HF_MODEL_CACHE = {}  # keyed by model name -> (tokenizer, model)
_HF_OUTLINES_CACHE = {}  # keyed by model name -> outlines model wrapper
_HF_LOCKS = {}  # keyed by model name -> threading.Lock


class APICallError(Exception):
    """Custom exception for API call failures"""

    pass


class BaseAgent:
    """Universal base agent for handling different LLM APIs.

    Configuration structure:
    {
        "provider": "openai|google|anthropic|sglang|ollama|together|openrouter",  # Required: API provider
        "model": "model-name",                                # Required: Model identifier
        "temperature": 0.7,                                   # Required: Temperature for sampling
        "max_retries": 3,                                     # Optional: Number of retries (default: 3)

        # Provider-specific configurations
        "project_id": "your-project",           # Required for Google
        "location": "us-central1",              # Required for Google
        "port": 30000,                          # Required for SGLang
        "base_url": "http://localhost:11434",   # Required for Ollama
        "api_key": "ollama",                    # Required for Ollama
        "response_format": {"type": "json_object"}  # Optional: For JSON responses (only for openai models) Make sure you include the word json in some form in the message
        "http_referer": "your-site-url",        # Optional for OpenRouter: Site URL for rankings
        "x_title": "your-site-name"             # Optional for OpenRouter: Site title for rankings
    }

    Usage:
        agent = BaseAgent(config)
        response = agent.call_api(messages, temperature=0.7)
        # Or with message inspection:
        response, messages = agent.call_api(messages, temperature=0.7, return_messages=True)
        # For JSON response (openai only):
        response = agent.call_api(messages, temperature=0.7, response_format={"type": "json_object"})
    """

    def __init__(self, config: Dict):
        """Initialize base agent with provider-specific setup."""
        self.config = config
        self.provider = config["provider"]
        self.max_retries = config.get("max_retries", 3)

        # Initialize client
        try:
            if self.provider == "openai":
                from openai import OpenAI
                import aisuite as ai

                if any(
                    f"o{i}" in config["model"] for i in range(1, 6)
                ):  # handles o1, o2, o3, o4, o5
                    self.client = OpenAI()
                    self.model = config["model"]
                else:
                    self.client = ai.Client()
                    self.model = f"{self.provider}:{config['model']}"
            elif self.provider == "openrouter":
                from openai import OpenAI

                # Initialize OpenRouter using OpenAI client with custom base URL
                self.client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENROUTER_API_KEY", ""),
                )
                self.model = config["model"]
                # Store optional headers for OpenRouter
                self.http_referer = config.get("http_referer")
                self.x_title = config.get("x_title")
            elif self.provider == "google":
                import aisuite as ai

                if "meta" in config["model"]:
                    self.project_id = config["project_id"]
                    self.location = config["location"]
                    self.model = config["model"]
                else:
                    os.environ["GOOGLE_PROJECT_ID"] = config["project_id"]
                    os.environ["GOOGLE_REGION"] = config["location"]
                    self.client = ai.Client()
                    self.model = f"{self.provider}:{config['model']}"
            elif self.provider == "transformers":
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch

                model_name = config["model"]
                if model_name not in _HF_MODEL_CACHE:
                    # Pick dtype: bfloat16 for CUDA/MPS, float32 for CPU
                    # float16 on CPU produces NaN logits and crashes during sampling
                    if torch.cuda.is_available():
                        dtype = torch.bfloat16
                        device_map = "auto"
                    elif torch.backends.mps.is_available():
                        dtype = torch.float32  # MPS float16 can also be unstable
                        device_map = "mps"
                    else:
                        dtype = torch.float32
                        device_map = "cpu"
                    logging.info(
                        f"[transformers] Loading model '{model_name}' | dtype={dtype} | device_map={device_map}"
                    )
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=dtype,
                        device_map=device_map,
                    )
                    logging.info(
                        f"[transformers] Model loaded | device={model.device} | params={sum(p.numel() for p in model.parameters()) / 1e9:.1f}B"
                    )
                    _HF_MODEL_CACHE[model_name] = (tokenizer, model)
                    _HF_LOCKS[model_name] = threading.Lock()
                    # Create Outlines wrapper for structured generation
                    import outlines
                    _HF_OUTLINES_CACHE[model_name] = outlines.from_transformers(
                        model, tokenizer
                    )
                    logging.info(f"[transformers] Outlines wrapper created for '{model_name}'")
                self.tokenizer, self.hf_model = _HF_MODEL_CACHE[model_name]
                self.outlines_model = _HF_OUTLINES_CACHE[model_name]
                self.model = model_name
            elif self.provider == "sglang":
                from openai import OpenAI

                # Initialize SGLang using OpenAI client
                self.client = OpenAI(
                    base_url=f"http://localhost:{config.get('port', 30000)}/v1",
                    api_key="None",  # SGLang doesn't require an API key
                )
                self.model = config["model"]
            elif self.provider == "ollama":
                import aisuite as ai

                self.client = ai.Client()
                self.model = f"{self.provider}:{config['model']}"
                self.client.configure(
                    {
                        "ollama": {
                            "timeout": 60,
                        }
                    }
                )
            elif self.provider == "together":
                import aisuite as ai

                self.client = ai.Client()
                self.model = f"{self.provider}:{config['model']}"
                self.client.configure(
                    {
                        "together": {
                            "timeout": 90,
                        }
                    }
                )
            else:
                import aisuite as ai

                # For all other providers
                self.client = ai.Client()
                self.model = f"{self.provider}:{config['model']}"

        except Exception as e:
            raise APICallError(f"Error initializing {self.provider} client: {str(e)}")

    def call_api(
        self,
        messages: List[Dict],
        temperature: float,
        response_format: Optional[Dict] = None,
        return_messages: bool = False,
    ) -> Union[str, Tuple[str, List[Dict]]]:
        """Universal API call handler with retries.

        Args:
            messages: List of message dictionaries
            temperature: Float value for temperature
            response_format: Optional response format specifications
            return_messages: If True, returns tuple of (response, messages)

        Returns:
            Either string response or tuple of (response, messages) if return_messages=True
        """
        print(
            f"{Fore.GREEN}Model is {self.model}, temperature is {temperature}{Style.RESET_ALL}"
        )

        # breakpoint()

        # Provider-specific configurations
        provider_configs = {
            "google": {"base_delay": 1, "retry_delay": 3, "jitter": 2},
            "openai": {"base_delay": 1, "retry_delay": 3, "jitter": 1},
            "together": {"base_delay": 1, "retry_delay": 3, "jitter": 1},
            "anthropic": {"base_delay": 1, "retry_delay": 2, "jitter": 1},
            "ollama": {"base_delay": 0, "retry_delay": 0, "jitter": 0},
            "transformers": {"base_delay": 0, "retry_delay": 0, "jitter": 0},
            "sglang": {"base_delay": 0, "retry_delay": 1, "jitter": 0.5},
            "openrouter": {
                "base_delay": 1,
                "retry_delay": 3,
                "jitter": 1,
            },  # Add OpenRouter configuration
        }

        config = provider_configs[self.provider]

        # print(json.dumps(messages, indent=2, ensure_ascii=False))

        for attempt in range(self.max_retries):
            try:
                # Add retry delay if needed
                if attempt > 0:
                    delay = config["retry_delay"] * attempt + uniform(
                        0, config["jitter"]
                    )
                    print(
                        f"\nRetry attempt {attempt + 1}/{self.max_retries}. Waiting {delay:.2f}s..."
                    )
                    sleep(delay)

                # Get response based on provider
                if self.provider == "openai":
                    if any(
                        f"o{i}" in self.model for i in range(1, 6)
                    ):  # handles o1, o2, o3, o4, o5
                        response = self._call_openai_o1_model(messages)
                    else:
                        api_params = {
                            "model": self.model,
                            "messages": messages,
                            "temperature": temperature,
                        }
                        if response_format:
                            api_params["response_format"] = response_format

                        response = self.client.chat.completions.create(**api_params)
                        response = response.choices[0].message.content
                elif self.provider == "openrouter":
                    # Setup extra headers for OpenRouter if provided
                    extra_headers = {}
                    if self.http_referer:
                        extra_headers["HTTP-Referer"] = self.http_referer
                    if self.x_title:
                        extra_headers["X-Title"] = self.x_title

                    api_params = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                    }
                    if response_format:
                        api_params["response_format"] = response_format
                    if extra_headers:
                        api_params["extra_headers"] = extra_headers

                    response = self.client.chat.completions.create(**api_params)
                    response = response.choices[0].message.content
                elif self.provider == "google" and "meta" in self.model:
                    response = self._call_google_meta_api(messages, temperature)
                elif self.provider == "transformers":
                    response = self._call_transformers(
                        messages, temperature, response_format=response_format
                    )
                elif self.provider == "sglang":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=2048,
                    )
                    response = response.choices[0].message.content
                else:
                    api_params = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                    }
                    if response_format:
                        api_params["response_format"] = response_format

                    response = self.client.chat.completions.create(**api_params)
                    response = response.choices[0].message.content

                # Log response preview
                logging.info(
                    f"[{self.provider}] Response preview: {response[:300]}..."
                    if len(response) > 300 else f"[{self.provider}] Response: {response}"
                )

                # Return based on return_messages flag
                return (response, messages) if return_messages else response

            except Exception as e:
                error_msg = str(e)
                if hasattr(e, "response"):
                    error_msg = f"Error code: {e.status_code} - {error_msg} - Response: {e.response}"

                logging.error(
                    f"BaseAgent: API call failed for {self.provider} (Attempt {attempt + 1}/{self.max_retries})",
                    exc_info=e,
                )

                if attempt == self.max_retries - 1:
                    # hack to prevent Anthropic error 400s from crashing the program
                    if self.provider == "anthropic" and e.status_code == 400:
                        return (
                            ("Sorry, I can't assist with that request.", messages)
                            if return_messages
                            else "Sorry, I can't assist with that request."
                        )
                    if self.provider == "google":
                        try:
                            from vertexai.generative_models._generative_models import ResponseValidationError
                            is_validation_error = isinstance(e, ResponseValidationError)
                        except ImportError:
                            is_validation_error = False
                    else:
                        is_validation_error = False
                    if is_validation_error:
                        return (
                            ("Sorry, I can't assist with that request.", messages)
                            if return_messages
                            else "Sorry, I can't assist with that request."
                        )

                    raise APICallError(
                        f"BaseAgent: Failed to get response from {self.provider}: {error_msg}"
                    )
                continue

    def _call_google_meta_api(self, messages: List[Dict], temperature: float) -> str:
        """Handle Google-hosted Meta models."""
        import google.auth
        import google.auth.transport.requests

        credentials, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)

        response = requests.post(
            f"https://{self.location}-aiplatform.googleapis.com/v1/projects/{self.project_id}/locations/{self.location}/endpoints/openapi/chat/completions",
            headers={
                "Authorization": f"Bearer {credentials.token}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
            },
        )

        if response.status_code != 200:
            raise APICallError(f"Error {response.status_code}: {response.text}")
        return response.json()["choices"][0]["message"]["content"]

    def _call_transformers(
        self,
        messages: List[Dict],
        temperature: float,
        response_format: Optional[Dict] = None,
    ) -> str:
        """Handle local HuggingFace transformers models."""
        import json as _json
        import re
        import torch

        # If a Pydantic schema is provided, use Outlines structured generation
        if response_format and "schema" in response_format:
            return self._call_transformers_structured(
                messages, temperature, response_format["schema"]
            )

        with _HF_LOCKS[self.model]:
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(input_text, return_tensors="pt").to(
                self.hf_model.device
            )
            input_len = inputs["input_ids"].shape[1]

            logging.info(
                f"[transformers] Generating | model={self.model} | input_tokens={input_len} "
                f"| temp={temperature} | device={self.hf_model.device} | dtype={self.hf_model.dtype}"
            )

            with torch.no_grad():
                outputs = self.hf_model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                )

            output_len = outputs[0].shape[0] - input_len
            # Decode only the newly generated tokens
            response = self.tokenizer.decode(
                outputs[0][input_len:], skip_special_tokens=True
            )

        # Strip <think>...</think> blocks from thinking models (e.g. Qwen3-*-Thinking)
        response = re.sub(r"<think>.*?</think>\s*", "", response, flags=re.DOTALL)

        logging.info(
            f"[transformers] Done | output_tokens={output_len} | response_chars={len(response)}"
        )
        if len(response.strip()) == 0:
            logging.warning(
                f"[transformers] Empty response after decoding! Raw output tokens: "
                f"{self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=False)[:200]}"
            )

        return response

    def _call_transformers_structured(
        self,
        messages: List[Dict],
        temperature: float,
        schema,
    ) -> str:
        """Generate structured JSON output using Outlines constrained generation."""
        import json as _json
        import outlines

        logging.info(
            f"[transformers/outlines] Structured generation | model={self.model} | "
            f"schema={schema.__name__} | temp={temperature}"
        )

        generator = outlines.Generator(self.outlines_model, output_type=schema)

        with _HF_LOCKS[self.model]:
            input_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            result = generator(
                input_text,
                max_new_tokens=4096,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
            )

        if hasattr(result, 'model_dump'):
            response = _json.dumps(result.model_dump(), indent=2)
        elif isinstance(result, str):
            response = result
        else:
            response = _json.dumps(result, indent=2)

        logging.info(
            f"[transformers/outlines] Done | response_chars={len(response)}"
        )

        return response

    def _call_openai_o1_model(self, messages: List[Dict]) -> str:
        """Handle OpenAI o1 models which only accept user messages."""
        # Warning about message handling
        print(
            "\nWarning: OpenAI o1 model only accepts user messages. System messages will be ignored."
        )

        # Format message for o1 model
        formatted_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": messages[-1][
                            "content"
                        ],  # Just take the last user message
                    }
                ],
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model, messages=formatted_messages
        )
        return response.choices[0].message.content


def main():
    """Test different model configurations."""
    test_configs = [
        # OpenAI standard config
        {
            "provider": "openai",
            "model": "gpt-4o",
            "max_retries": 3,
            "temperature": 0,
            # "response_format": {"type": "json_object"}  # Correct format as a dictionary if json response is needed (only for openai models) Make sure you include the word json in some form in the message.
        },
        # SGLang config
        {
            "provider": "sglang",
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "port": 30000,
            "max_retries": 3,
            "temperature": 0,
        },
        # OpenAI preview model config
        {
            "provider": "openai",
            "model": "o1-preview",
            "max_retries": 3,
            "temperature": 0,
        },
        # Google standard config
        {
            "provider": "google",
            "model": "gemini-1.5-pro-002",
            "project_id": "mars-lab-429920",
            "location": "us-central1",
            "max_retries": 3,
            "temperature": 0,
        },
        # Google-hosted Meta model config
        {
            "provider": "google",
            "model": "meta/llama-3.1-405b-instruct-maas",
            "project_id": "mars-lab-429920",
            "location": "us-central1",
            "max_retries": 3,
            "temperature": 0,
        },
        # SGLang config
        {
            "provider": "sglang",
            "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "port": 30000,
            "max_retries": 3,
            "temperature": 0,
        },
        # Ollama config
        {
            "provider": "ollama",
            "model": "llama3.1:latest",
            "base_url": "http://localhost:11434/v1",
            "api_key": "ollama",
            "max_retries": 3,
            "temperature": 0,
        },
        # Anthropic config
        {
            "provider": "anthropic",
            "model": "claude-3-sonnet-20240229",
            "max_retries": 3,
            "temperature": 0,
        },
        # OpenRouter config
        # other model anthropic/claude-3, anthropic/claude-3.7-sonnet, deepseek/deepseek-chat
        {
            "provider": "openrouter",
            "model": "deepseek/deepseek-r1",
            "max_retries": 3,
            "temperature": 0,
            # "http_referer": "https://yourapp.com",  # Optional
            # "x_title": "Your App Name"  # Optional
        },
    ]

    # Test message for all models
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    for config in test_configs:
        print("\n" + "=" * 50)
        print(f"Testing {config['provider']} with model {config['model']}")
        print("=" * 50)

        try:
            agent = BaseAgent(config)
            # Test with message inspection
            response, messages = agent.call_api(
                messages=test_messages,
                temperature=config["temperature"],
                # response_format=config.get('response_format'),  # Pass response_format from config
                return_messages=True,
            )
            print(f"\nMessages sent: {messages}")
            print(f"\nResponse: {response}")
            print("\nTest completed successfully!")

        except Exception as e:
            print(f"Error testing configuration: {str(e)}")

        print("-" * 50)
    breakpoint()


if __name__ == "__main__":
    main()
