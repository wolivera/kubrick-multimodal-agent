import uuid
from datetime import datetime

from groq import Groq

from agent_api.config import settings
from agent_api.core.agent.base_agent import BaseAgent
from agent_api.core.agent.memory import Memory, MemoryRecord


class GroqAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str,
        mcp_server: str,
        memory: Memory = None,
    ):
        super().__init__(name, system_prompt, model, mcp_server, memory)
        self.client = Groq(api_key=settings.GROQ_API_KEY)

    def generate_response(self, message: str) -> str:
        chat_history = [
            {"role": "system", "content": self.system_prompt},
            *[
                {"role": record.role, "content": record.content}
                for record in self.memory.get_latest(n=settings.AGENT_MEMORY_SIZE)
            ],
        ]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                *chat_history,
                {"role": "user", "content": message},
            ],
        )
        return response.choices[0].message.content

    def chat(self, message: str):
        self.memory.insert(
            MemoryRecord(
                message_id=str(uuid.uuid4()),
                role="user",
                content=message,
                timestamp=datetime.now(),
            )
        )

        response = self.generate_response(message)

        self.memory.insert(
            MemoryRecord(
                message_id=str(uuid.uuid4()),
                role="assistant",
                content=response,
                timestamp=datetime.now(),
            )
        )

        return response
