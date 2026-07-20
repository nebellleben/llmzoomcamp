"""OpenAI Responses-API adapter for Zhipu (GLM) via the coding-plan endpoint.

The homework starter (`rag_helper.RAGBase.llm`) calls::

    self.llm_client.responses.create(model=..., input=messages)

and then reads `response.output_text` and `response.usage.input_tokens` /
`response.usage.output_tokens`.  Zhipu does not implement the OpenAI
Responses API, so this adapter wraps a normal OpenAI-compatible
`chat.completions` call and exposes the small Responses-style surface the
starter needs.

Default endpoint is the "coding plan" subscription endpoint, which is what
the `zhipuai-coding-plan` key authorizes. The default model is `glm-4-flash`,
a non-reasoning model with clean token counts (closest analogue to
`gpt-5.4-mini`).
"""

import os

from openai import OpenAI

ZHIPU_CODING_BASE_URL = "https://open.bigmodel.cn/api/coding/paas/v4"
DEFAULT_MODEL = "glm-4-flash"


class _Usage:
    """Mimics the relevant subset of OpenAI Responses `response.usage`."""

    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.input_tokens = prompt_tokens
        self.output_tokens = completion_tokens


class _Response:
    """Mimics the relevant subset of an OpenAI Responses response object."""

    def __init__(self, chat_completion):
        self.output_text = chat_completion.choices[0].message.content or ""
        u = chat_completion.usage
        self.usage = _Usage(
            prompt_tokens=getattr(u, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(u, "completion_tokens", 0) or 0,
        )
        self._raw = chat_completion


class _ResponsesNamespace:
    """`client.responses` namespace shim with the `.create()` method."""

    def __init__(self, client: OpenAI, model: str):
        self._client = client
        self._model = model

    def create(self, *, model=None, input=None, **kwargs):
        messages = []
        for m in input or []:
            role = m.get("role", "user")
            if role == "developer":
                role = "system"
            messages.append({"role": role, "content": m.get("content", "")})

        resp = self._client.chat.completions.create(
            model=model or self._model,
            messages=messages,
        )
        return _Response(resp)


class ZhipuResponsesClient:
    """Drop-in replacement for `OpenAI()` for code using the Responses API.

    Usage::

        from zhipu_client import ZhipuResponsesClient
        rag = RAGBase(index=index, llm_client=ZhipuResponsesClient())
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = DEFAULT_MODEL,
    ):
        api_key = (
            api_key
            or os.environ.get("ZHIPU_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "No Zhipu API key found. Set ZHIPU_API_KEY (or OPENAI_API_KEY)."
            )
        base_url = base_url or os.environ.get("ZHIPU_BASE_URL") or ZHIPU_CODING_BASE_URL
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self.responses = _ResponsesNamespace(self._client, model)

    @property
    def model(self) -> str:
        return self._model
