from abc import ABC, abstractmethod

from fastmcp import Client
from fastmcp.types import Tool
from loguru import logger

from agent_api.agent.memory import Memory


class BaseAgent(ABC):
    """
    Base class for all agents.
    """

    def __init__(
        self,
        name: str,
        routing_system_prompt: str,
        tool_use_system_prompt: str,
        general_system_prompt: str,
        model: str,
        mcp_server: str,
        memory: Memory = None,
    ):
        self.name = name
        self.routing_system_prompt = routing_system_prompt
        self.tool_use_system_prompt = tool_use_system_prompt
        self.general_system_prompt = general_system_prompt
        self.model = model
        self.mcp_client = Client(mcp_server)
        self.memory = memory if memory else Memory(name)

    def reset_memory(self):
        self.memory.reset_memory()
        
    def filter_active_tools(self, tools: list[Tool]) -> list[Tool]:
        """
        Filter the list of tools to only include the active tools.
        """
        return [tool for tool in tools if tool.name in self.active_tools]

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
                tools = self.filter_active_tools(tools)
                logger.info(f"Filtered tools to {len(tools)} active tools")
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
