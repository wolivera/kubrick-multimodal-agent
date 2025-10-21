import click
from fastmcp import FastMCP

from kubrick_mcp.prompts import general_system_prompt, routing_system_prompt, tool_use_system_prompt
from kubrick_mcp.resources import list_tables
from kubrick_mcp.tools import (
    ask_question_about_video,
    get_video_clip_from_image,
    get_video_clip_from_user_query,
    process_video,
)


def add_mcp_resources(mcp: FastMCP):
    mcp.add_resource_fn(
        fn=list_tables,
        uri="file:///app/.records/records.json",
        name="list_tables",
        description="List all video indexes currently available.",
        tags={"resource", "all"},
    )


def add_mcp_prompts(mcp: FastMCP):
    @mcp.prompt(
        name="routing_system_prompt",
        description="Latest version of the routing prompt from Opik.",
    )
    def routing_prompt() -> str:
        return routing_system_prompt()

    @mcp.prompt(
        name="tool_use_system_prompt",
        description="Latest version of the tool use prompt from Opik.",
    )
    def tool_use_prompt() -> str:
        return tool_use_system_prompt()

    @mcp.prompt(
        name="general_system_prompt",
        description="Latest version of the general prompt from Opik.",
    )
    def general_prompt() -> str:
        return general_system_prompt()


mcp = FastMCP("VideoProcessor")

# Register tools using decorator pattern
mcp.tool(process_video)
mcp.tool(get_video_clip_from_user_query)
mcp.tool(get_video_clip_from_image)
mcp.tool(ask_question_about_video)

add_mcp_prompts(mcp)
add_mcp_resources(mcp)


@click.command()
@click.option("--port", default=9090, help="FastMCP server port")
@click.option("--host", default="0.0.0.0", help="FastMCP server host")
@click.option("--transport", default="streamable-http", help="MCP Transport protocol type")
def run_mcp(port, host, transport):
    """
    Run the FastMCP server with the specified port, host, and transport protocol.
    """
    mcp.run(host=host, port=port, transport=transport)


if __name__ == "__main__":
    run_mcp()
