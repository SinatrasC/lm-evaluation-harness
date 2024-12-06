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
        **kwargs,
    ):
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            model=model,
            max_length=max_length,
            max_gen_toks=max_gen_toks,
            batch_size=batch_size,
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
        Create the payload for the chat endpoint. We assume --apply_chat_template 
        has already formatted the `messages` as a list of dicts like:
        [{"role": "system", "content": ...}, {"role": "user", "content": ...}, ...]

        Our local endpoint expects similar arguments as OpenAI:
        {
            "messages": messages,
            "model": self.model,
            "max_tokens": ...,
            "temperature": ...,
            "stop": [...]
        }
        """
        gen_kwargs = gen_kwargs or {}
        gen_kwargs.pop("do_sample", False)
        # max_tokens logic
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)

        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", [eos]), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]

        payload = {
            "messages": messages,
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop[:4],
        }
        # Add any remaining gen_kwargs to payload
        payload.update(gen_kwargs)
        return payload

    def _stream_completion(self, payload: dict) -> Dict:
        """
        Handle the streaming response from the local endpoint.
        The endpoint returns `text/event-stream`, with lines like:
        data: {"id":"...", "choices":[{"delta":{"content":"..."},"finish_reason":null}]}

        We'll accumulate `content` until finish_reason=stop or the stream ends.
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
                        # Ignore malformed lines
                        continue

                    # Each chunk can have multiple choices; we assume one choice.
                    if "choices" in data and data["choices"]:
                        for choice in data["choices"]:
                            delta = choice.get("delta", {})
                            if "role" in delta and not role_set:
                                # The first delta often sets the role
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
                # If we reach here, we got [DONE] or ended stream without a stop.
                return {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": full_response_text
                        }
                    }]
                }

        except RequestException as e:
            raise ValueError(f"Error contacting local model: {e}")

    def model_call(self, payload):
        """
        Override model_call to handle streaming response.
        We'll return a dictionary similar to OpenAI's final JSON (no streaming).
        """
        return self._stream_completion(payload)

    @staticmethod
    def parse_logprobs(
        outputs: Union[Dict, List[Dict]],
        tokens: List[List[int]] = None,
        ctxlens: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        """
        Chat completions do not provide token-level logprobs for prompts.
        We raise NotImplementedError here.
        """
        raise NotImplementedError(
            "Loglikelihood is not supported for chat completions."
        )

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        """
        Parse the generated assistant message from the final returned dict.
        """
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
        # For chat models, we return the input as-is if needed.
        # Typically, chat messages are passed as lists of dicts and not tokenized here.
        return string

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood is not supported for chat completions."
        )
