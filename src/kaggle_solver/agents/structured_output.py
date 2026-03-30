from __future__ import annotations

import json
from typing import Sequence

from autogen_agentchat.messages import BaseAgentEvent, BaseChatMessage, ToolCallExecutionEvent, ToolCallRequestEvent
from autogen_core.tools import FunctionTool
from pydantic import BaseModel

FINAL_STRUCTURED_RESPONSE_TOOL_NAME = "submit_structured_response"


def build_structured_response_tool(output_model: type[BaseModel]) -> FunctionTool:
    async def submit_structured_response(payload) -> str:
        output_model.model_validate(payload)
        return f"Accepted final structured response for {output_model.__name__}."

    submit_structured_response.__annotations__ = {
        "payload": output_model,
        "return": str,
    }
    submit_structured_response.__name__ = FINAL_STRUCTURED_RESPONSE_TOOL_NAME
    submit_structured_response.__doc__ = (
        f"Submit the final response as a payload matching the {output_model.__name__} schema. "
        "Call this exactly once as the final action."
    )

    return FunctionTool(
        submit_structured_response,
        description=(
            f"Submit the final response as a payload matching the {output_model.__name__} schema. "
            "Call this exactly once as the final action."
        ),
        name=FINAL_STRUCTURED_RESPONSE_TOOL_NAME,
        strict=True,
    )


def extract_structured_response_payload(
    inner_messages: Sequence[BaseAgentEvent | BaseChatMessage] | None,
    output_model: type[BaseModel],
) -> BaseModel | None:
    if not inner_messages:
        return None

    successful_call_ids: set[str] = set()
    requested_calls: list[tuple[str, str]] = []

    for message in inner_messages:
        if isinstance(message, ToolCallExecutionEvent):
            for result in message.content:
                if result.name == FINAL_STRUCTURED_RESPONSE_TOOL_NAME and not result.is_error:
                    successful_call_ids.add(result.call_id)
        elif isinstance(message, ToolCallRequestEvent):
            for call in message.content:
                if call.name == FINAL_STRUCTURED_RESPONSE_TOOL_NAME:
                    requested_calls.append((call.id, call.arguments))

    for call_id, arguments in reversed(requested_calls):
        if call_id and call_id not in successful_call_ids:
            continue
        parsed_arguments = json.loads(arguments)
        payload = parsed_arguments.get("payload", parsed_arguments)
        return output_model.model_validate(payload)

    return None
