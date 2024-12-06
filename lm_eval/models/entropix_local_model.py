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
    OpenAI's chat completion streaming API.
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
        
        if self._batch_size > 1:
            eval_logger.warning(
                "Chat completions does not support batching. Defaulting to batch size 1."
            )
            self._batch_size = 1

    def _extract_prompt(self, obj: Any) -> Optional[str]:
        """Extract prompt from various types of objects"""
        if hasattr(obj, 'prompt'):
            return obj.prompt
        elif isinstance(obj, (str, int, float)):
            return str(obj)
        elif isinstance(obj, (list, tuple)):
            # For containers, try each element
            for item in obj:
                result = self._extract_prompt(item)
                if result is not None:
                    return result
        return None

    def _parse_messages(self, messages: Any) -> List[Dict[str, str]]:
        """
        Parse messages from various input formats into the expected format.
        Now handles nested structures and multiple JsonChatStr objects.
        """
        try:
            # Extract prompt if it exists at any level
            prompt = self._extract_prompt(messages)
            if prompt:
                try:
                    # Try parsing as JSON first
                    parsed = json.loads(prompt)
                    if isinstance(parsed, list):
                        return parsed
                    elif isinstance(parsed, dict):
                        return [parsed]
                except json.JSONDecodeError:
                    # If not JSON, treat as user message
                    return [{"role": "user", "content": prompt}]

            # Handle direct containers
            if isinstance(messages, (tuple, list)):
                messages = list(messages)
                if len(messages) == 1:
                    return self._parse_messages(messages[0])
                return messages

            # Handle dictionaries
            if isinstance(messages, dict):
                return [messages]

            # Handle strings
            if isinstance(messages, str):
                try:
                    parsed = json.loads(messages)
                    if isinstance(parsed, list):
                        return parsed
                    elif isinstance(parsed, dict):
                        return [parsed]
                except json.JSONDecodeError:
                    return [{"role": "user", "content": messages}]

            raise ValueError(f"Unable to parse messages from: {type(messages)}")
        except Exception as e:
            raise ValueError(f"Failed to parse messages: {e}") from e

    def _validate_message(self, msg: Dict) -> Dict[str, str]:
        """Validate and format a single message"""
        if not isinstance(msg, dict):
            raise ValueError(f"Message must be a dictionary, got {type(msg)}")
        
        if 'role' not in msg or 'content' not in msg:
            raise ValueError(f"Message missing required fields: {msg}")
            
        if msg['role'] not in ['system', 'user', 'assistant']:
            raise ValueError(f"Invalid role: {msg['role']}")
            
        return {
            "role": msg['role'],
            "content": str(msg['content'])
        }

    def _create_payload(
        self,
        messages: Any,
        generate=False,
        gen_kwargs: Optional[dict] = None,
        seed=1234,
        eos="<|endoftext|>",
        **kwargs,
    ) -> dict:
        """Create the payload for the chat endpoint"""
        gen_kwargs = gen_kwargs or {}
        
        # Parse and validate messages
        try:
            parsed_messages = self._parse_messages(messages)
            formatted_messages = [self._validate_message(msg) for msg in parsed_messages]
        except Exception as e:
            raise ValueError(f"Message processing failed: {e}")

        # Handle parameters
        max_tokens = gen_kwargs.pop("max_tokens", None) or gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        
        # Handle stop sequences
        stop = handle_stop_sequences(gen_kwargs.pop("until", [eos]), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]

        # Create base payload
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": True,
            "stop": stop[:4]
        }

        # Add additional parameters
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
        """Handle streaming response from the endpoint"""
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
                
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": full_response_text
                        }
                    }]
                }

        except RequestException as e:
            if hasattr(e, 'response'):
                if e.response.status_code == 422:
                    error_detail = e.response.json().get('detail', 'Unknown validation error')
                    raise ValueError(f"Request validation failed: {error_detail}")
                elif e.response.status_code == 503:
                    raise ValueError("Model not initialized or unavailable")
            raise ValueError(f"Error contacting local model: {e}")

    def model_call(self, **kwargs) -> Dict:
        """Handle calls from lm_eval"""
        messages = kwargs.pop("messages", None)
        if messages is None:
            raise ValueError("No messages provided to model_call")
        
        gen_kwargs = kwargs.pop("gen_kwargs", {})
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
        raise NotImplementedError("Loglikelihood not supported for chat completions")

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        """Parse the generated assistant message"""
        if not isinstance(outputs, list):
            outputs = [outputs]
        res = []
        for out in outputs:
            for choice in out.get("choices", []):
                msg = choice.get("message", {})
                content = msg.get("content", "")
                res.append(content)
        return res

    def tok_encode(self, string: Union[str, Any], **kwargs) -> Union[List[str], List[int], Any]:
        """Return input as-is for chat models"""
        return string

    def loglikelihood(self, requests, **kwargs):
        """Chat completions do not support loglikelihood computation"""
        raise NotImplementedError("Loglikelihood not supported for chat completions")
