import json

from typing import Optional, Dict, Union

from haystack.nodes import PromptNode as HaystackPromptNode
from haystack.nodes.prompt import PromptTemplate as HaystackPromptTemplate

from .base import Node

try:
    import outlines.models as models
    import outlines.text.generate as generate

    from pydantic import BaseModel

    import torch
except ImportError as exc:
    raise ImportError("Try 'pip install outlines'") from exc


class GuidedPromptNode(Node):
    def __init__(
        self,
        model_name_or_path: str,
        schema: Union[str, BaseModel],
        max_length: int = 100,
        device: Optional[str] = None,
        model_kwargs: Dict = None,
        manual_seed: Optional[int] = None,
    ) -> None:
        self.max_length = max_length
        model_kwargs = model_kwargs or {}
        self._model = models.transformers(model_name_or_path, device=device, **model_kwargs)
        # JSON schema of class
        if not isinstance(schema, str):
            schema = json.dumps(schema.schema())

        self._generator = generate.json(
            self._model,
            schema,
            self.max_length,
        )

        self.rng = torch.Generator(device=device)
        if manual_seed is not None:
            self.rng.manual_seed(manual_seed)

    def run(self, prompt_template: HaystackPromptTemplate, **kwargs):
        return self._generator(prompt_template.prompt_text, rng=self.rng)
