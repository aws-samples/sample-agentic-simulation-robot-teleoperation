"""Bedrock model wrapper that keeps camera frames visible to models which reject images inside tool results.

Measured 2026-09-28 against ``global.openai.gpt-6-astra`` (GPT-6 Astra on Bedrock): an ``image``
block INSIDE a ``toolResult`` fails with ``ValidationException: This model doesn't support the image
field for user messages``, on both Converse and ConverseStream, while the same image as a sibling
block of the tool result in the same user turn is accepted (and so is an image in a plain user turn).
Anthropic models accept both.

Strands emits every tool return as a toolResult block, so the ``look`` tool's frame would kill the agent
loop on Astra at its first call (observed on the OpenArm agent, 2026-09-28). This subclass rewrites the request: images found in toolResult
content are replaced by a short text pointer and re-emitted as sibling ``image`` blocks of the same
user message. The model still sees the frame; the wire format is the one the model accepts.
"""
from __future__ import annotations

from typing import Any

from strands.models import BedrockModel


def _hoist_tool_result_images(request: dict[str, Any]) -> int:
    moved = 0
    for msg in request.get("messages", []):
        if msg.get("role") != "user":
            continue
        new_content: list[dict[str, Any]] = []
        hoisted: list[dict[str, Any]] = []
        for block in msg.get("content", []):
            tr = block.get("toolResult")
            if not tr:
                new_content.append(block)
                continue
            kept: list[dict[str, Any]] = []
            for c in tr.get("content", []):
                if "image" in c:
                    hoisted.append({"image": c["image"]})
                    kept.append({"text": "[camera frame attached below this tool result]"})
                    moved += 1
                else:
                    kept.append(c)
            new_block = dict(block)
            new_block["toolResult"] = dict(tr, content=kept)
            new_content.append(new_block)
        if hoisted:
            msg["content"] = new_content + hoisted
    return moved


class ToolImageHoistingBedrockModel(BedrockModel):
    """BedrockModel that hoists images out of toolResult blocks (see module docstring)."""

    def format_request(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        request = super().format_request(*args, **kwargs)
        _hoist_tool_result_images(request)
        return request


def needs_hoisting(model_id: str) -> bool:
    """Only non-Anthropic providers on Bedrock have shown the restriction; Anthropic accepts both."""
    return "anthropic" not in model_id.lower()
