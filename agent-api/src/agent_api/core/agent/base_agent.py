from abc import ABC, abstractmethod

from fastmcp import Client
from loguru import logger

from agent_api.core.agent.memory import Memory


class BaseAgent(ABC):
    """
    Base class for all agents.
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: str,
        mcp_server: str,
        memory: Memory = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.mcp_client = Client(mcp_server)
        self.memory = memory if memory else Memory(name)

    def reset_memory(self):
        self.memory.reset_memory()

    async def discover_tools(self) -> list:
        """
        Discover and register available tools from the MCP server.

        This method connects to the MCP server and retrieves the list of available tools.
        Each tool contains metadata like name, description, and parameters.

        Returns:
            list: A list of Tool objects containing the discovered tools and their metadata

        Raises:
            ConnectionError: If unable to connect to the MCP server
            Exception: If tool discovery fails for any other reason
        """
        try:
            async with self.mcp_client as client:
                tools = await client.list_tools()
                if not tools:
                    logger.info("No tools were discovered from the MCP server")
                    return []
                logger.info(f"Discovered {len(tools)} tools:")
                for tool in tools:
                    logger.info(f"- {tool.name}: {tool.description}")

                return tools
        except ConnectionError as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise
        except Exception as e:
            logger.error(f"Tool discovery failed: {e}")
            raise

    @abstractmethod
    def chat(self, message: str) -> str:
        raise NotImplementedError("Chat is not implemented in the base class.")
