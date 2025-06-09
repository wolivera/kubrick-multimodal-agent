import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from loguru import logger

import instructor
from groq import Groq

from agent_api.agent.groq.groq_tool import transform_tool_definition
from agent_api.agent.groq.models import RoutingResponseModel
from agent_api.config import settings
from agent_api.core.agent.base_agent import BaseAgent
from agent_api.core.agent.memory import Memory, MemoryRecord

logger.bind(name="GroqAgent")


class GroqAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        routing_system_prompt: str,
        tool_use_system_prompt: str,
        general_system_prompt: str,
        model: str,
        mcp_server: str,
        memory: Optional[Memory] = None,
    ):
        super().__init__(name, routing_system_prompt, tool_use_system_prompt, general_system_prompt, model, mcp_server, memory)
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.instructor_client = instructor.from_groq(
            self.client, mode=instructor.Mode.JSON
        )
        self.tools = self._get_tools()

    def _get_tools(self) -> List[Dict[str, Any]]:
        """Transform and return the list of available tools."""
        return [transform_tool_definition(tool) for tool in self.discover_tools()]

    def _build_chat_history(self, system_prompt: str) -> List[Dict[str, str]]:
        """Build chat history with system prompt and recent memory records."""
        return [
            {"role": "system", "content": system_prompt},
            *[
                {"role": record.role, "content": record.content}
                for record in self.memory.get_latest(n=settings.AGENT_MEMORY_SIZE)
            ],
        ]

    def _route_query(self, message: str) -> bool:
        """Determine if the message requires tool usage."""
        response = self.instructor_client.chat.completions.create(
            model=settings.GROQ_ROUTING_MODEL,
            response_model=RoutingResponseModel,
            messages=[
                {"role": "system", "content": self.routing_system_prompt},
                {"role": "user", "content": message},
            ],
            max_completion_tokens=20,
        )
        return response.tool_use


    def _run_with_tool(self, message: str) -> str:
        """Execute chat completion with tool usage."""
        chat_history = self._build_chat_history(self.tool_use_system_prompt)

        response = self.client.chat.completions.create(
            model=settings.GROQ_TOOL_USE_MODEL,
            messages=chat_history,
            tools=self.tools,
            tool_choice="auto",
            max_completion_tokens=4096,
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls

        if not tool_calls:
            return response_message.content

        chat_history.append(response_message)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            function_response = await self.mcp_client.call_tool(
                function_name, function_args
            )
            chat_history.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )

        second_response = self.client.chat.completions.create(
            model=self.tool_use_prompt, messages=chat_history
        )
        return second_response.choices[0].message.content


    def _run_general(self, message: str) -> str:
        """Execute general chat completion without tool usage."""
        chat_history = self._build_chat_history(self.general_system_prompt)

        response = self.client.chat.completions.create(
            model=settings.GROQ_GENERAL_MODEL,
            messages=chat_history,
            max_completion_tokens=settings.GROQ_GENERAL_MODEL_MAX_TOKENS,
        )
        return response.choices[0].message.content

    def _add_to_memory(self, role: str, content: str) -> None:
        """Add a message to the agent's memory."""
        self.memory.insert(
            MemoryRecord(
                message_id=str(uuid.uuid4()),
                role=role,
                content=content,
                timestamp=datetime.now(),
            )
        )

    def chat(self, message: str) -> str:
        """Process a chat message and return the response."""
        self._add_to_memory("user", message)

        tool_use = self._route_query(message)
        
        if tool_use:
            response = self._run_with_tool(message)
        else:
            response = self._run_general(message)
        
        self._add_to_memory("assistant", response)
        return response
