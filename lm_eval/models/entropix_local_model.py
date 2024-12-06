import os
import json
import requests
from requests.exceptions import RequestException
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Union, Generator
from lm_eval.api.registry import register_model
from lm_eval.models.api_models import TemplateAPI
from lm_eval.models.utils import handle_stop_sequences
from lm_eval.utils import eval_logger

@register_model("entropix-local-chat")
class EntropixLocalChatModel(TemplateAPI):
    """
    A local model interface for a server that exposes an endpoint similar to 
    OpenAI's chat completion streaming API, as defined in server_main.py.

    This model:
    - Uses the /v1/chat/completions endpoint.
    - Expects a JSON body with `messages` and returns streamed responses.
    - Requires `--apply_chat_template` when running lm-eval to ensure the 
      requests are in a chat-compatible format.
    """

    def __init__(
        self,
        base_url="http://localhost:8000/v1/chat/completions",
        tokenizer_backend=None,
        tokenized_requests=False,
        model="entropix-chat",
        max_length=2048,
        max_gen_toks=2048,
        batch_size=1,
        timeout=30,
        verify_certificate=True,
        **kwargs,
    ):
        # Store attributes before calling super().__init__
        self._base_url = base_url
        self._timeout = timeout
        self._verify_certificate = verify_certificate
        
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            model=model,
            max_length=max_length,
            max_gen_toks=max_gen_toks,
            batch_size=batch_size,
            timeout=timeout,
            verify_certificate=verify_certificate,
            **kwargs,
        )
        
        # Chat completions typically do not support batching or logprobs.
        if self._batch_size > 1:
            eval_logger.warning(
                "Chat completions does not support batching. Defaulting to batch size 1."
            )
            self._batch_size = 1

    @cached_property
    def api_key(self):
        # If your local model doesn't require a key, return an empty string.
        # If it does, set an ENV variable and read it here.
        return os.environ.get("API_KEYS", "sk-test-key")

    def headers(self) -> Dict[str, str]:
        # Set headers if needed. The local model might not require any key.
        # If you have an auth token, include it here.
        headers = {
            "Content-Type": "application/json",
        }
        return headers

    def _parse_messages(self, messages) -> List[Dict]:
        """Parse messages from various input formats into the expected format"""
        try:
            # Handle JsonChatStr objects which have a prompt attribute containing a string
            if hasattr(messages, 'prompt'):
                try:
                    # Try to parse the prompt as a JSON string
                    return json.loads(messages.prompt)
                except json.JSONDecodeError:
                    # If not valid JSON, treat it as a single user message
                    return [{"role": "user", "content": messages.prompt}]

            # Handle direct list/dict input
            if isinstance(messages, list):
                return messages
            if isinstance(messages, dict):
                return [messages]
            
            # Handle string input by treating it as user message
            if isinstance(messages, str):
                return [{"role": "user", "content": messages}]
            
            raise ValueError(f"Unsupported message format: {type(messages)}")
        except Exception as e:
            raise ValueError(f"Failed to parse messages: {e}") from e

    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        eos="<|endoftext|>",
        **kwargs,
    ) -> dict:
        """
        Create the payload for the chat endpoint according to the server's ChatCompletionRequest format.
        Handles validation and formatting of messages and parameters.
        """
        gen_kwargs = gen_kwargs or {}
        
        # Parse and validate messages
        try:
            formatted_messages = self._parse_messages(messages)
        except Exception as e:
            raise ValueError(f"Failed to parse messages: {e}")

        # Validate each message
        for msg in formatted_messages:
            if not isinstance(msg, dict) or 'role' not in msg or 'content' not in msg:
                raise ValueError(f"Invalid message format after parsing: {msg}")
            if msg['role'] not in ['system', 'user', 'assistant']:
                raise ValueError(f"Invalid role in message: {msg['role']}")
            msg['content'] = str(msg['content'])

        # Handle max_tokens
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)

        # Handle temperature and other parameters
        temperature = gen_kwargs.pop("temperature", 0)
        
        # Format stop sequences
        stop = handle_stop_sequences(gen_kwargs.pop("until", [eos]), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]
        
        # Create the request payload according to server's expected format
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": float(temperature),  # Ensure temperature is float
            "max_tokens": int(max_tokens),      # Ensure max_tokens is int
            "stream": True,                     # Always stream for our implementation
            "stop": stop[:4]                    # OpenAI API limit
        }

        # Add any additional supported parameters from gen_kwargs
        supported_params = {
            "top_p": float,
            "frequency_penalty": float,
            "presence_penalty": float,
            "seed": int
        }
        
        for param, type_conv in supported_params.items():
            if param in gen_kwargs:
                payload[param] = type_conv(gen_kwargs[param])
        
        return payload

    def _stream_completion(self, payload: dict) -> Dict:
        """
        Handle the streaming response from the local endpoint.
        Includes proper error handling for validation errors and malformed responses.
        """
        try:
            with requests.post(
                self._base_url,
                headers=self.headers(),
                json=payload,
                stream=True,
                timeout=self._timeout,
                verify=self._verify_certificate
            ) as response:
                response.raise_for_status()
                full_response_text = ""
                role_set = False
                
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data: "):
                        continue
                        
                    data_str = line[len("data: "):].strip()
                    if data_str == "[DONE]":
                        break
                        
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Process choices in the response
                    if "choices" in data and data["choices"]:
                        for choice in data["choices"]:
                            delta = choice.get("delta", {})
                            if "role" in delta and not role_set:
                                role_set = True
                            if "content" in delta:
                                full_response_text += delta["content"]
                            if choice.get("finish_reason") == "stop":
                                return {
                                    "choices": [{
                                        "message": {
                                            "role": "assistant",
                                            "content": full_response_text
                                        }
                                    }]
                                }
                
                # Return accumulated response if stream ends without explicit stop
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": full_response_text
                        }
                    }]
                }

        except RequestException as e:
            if e.response is not None:
                if e.response.status_code == 422:
                    error_detail = e.response.json().get('detail', 'Unknown validation error')
                    raise ValueError(f"Request validation failed: {error_detail}")
                elif e.response.status_code == 503:
                    raise ValueError("Model not initialized or unavailable")
            raise ValueError(f"Error contacting local model: {e}")

    def model_call(self, **kwargs):
        """Handle calls from lm_eval that provide `messages` and `gen_kwargs`"""
        messages = kwargs.pop("messages", None)
        gen_kwargs = kwargs.pop("gen_kwargs", {})
    
        if messages is None:
            raise ValueError("No messages provided to model_call")
    
        payload = self._create_payload(
            messages=messages,
            generate=True,
            gen_kwargs=gen_kwargs,
        )
    
        return self._stream_completion(payload)

    @staticmethod
    def parse_logprobs(
        outputs: Union[Dict, List[Dict]],
        tokens: List[List[int]] = None,
        ctxlens: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        """Chat completions do not support token-level logprobs"""
        raise NotImplementedError(
            "Loglikelihood is not supported for chat completions."
        )

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        """Parse the generated assistant message from the final returned dict"""
        if not isinstance(outputs, list):
            outputs = [outputs]
        res = []
        for out in outputs:
            for choice in out.get("choices", []):
                msg = choice.get("message", {})
                content = msg.get("content", "")
                res.append(content)
        return res

    def tok_encode(
        self,
        string: Union[str, Any],
        left_truncate_len=None,
        add_special_tokens=None,
        **kwargs,
    ) -> Union[List[str], List[int], Any]:
        """Return input as-is for chat models"""
        return string

    def loglikelihood(self, requests, **kwargs):
        """Chat completions do not support loglikelihood computation"""
        raise NotImplementedError(
            "Loglikelihood is not supported for chat completions."
        )
