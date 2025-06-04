from fastmcp import FastMCP

from .prompts import struct_output_prompt, system_prompt
from .resources import list_table_info, list_tables
from .tools import (
    get_clip_by_caption_sim,
    get_clip_by_image_sim,
    get_clip_by_speech_sim,
    process_video,
)


def add_mcp_tools(mcp: FastMCP):
    mcp.add_tool(
        name="process_video",
        description="Process a video file and prepare it for searching.",
        fn=process_video,
        tags={"video", "process"},
    )

    mcp.add_tool(
        name="get_clip_by_speech_sim",
        description="Get a video clip based on a user query using the transcripts index.",
        fn=get_clip_by_speech_sim,
        tags={"video", "search", "transcript"},
    )

    mcp.add_tool(
        name="get_clip_by_image_sim",
        description="Get a video clip based on a user query using the image index.",
        fn=get_clip_by_image_sim,
        tags={"video", "search", "image"},
    )

    mcp.add_tool(
        name="get_clip_by_caption_sim",
        description="Get a video clip based on a user query using the caption index.",
        fn=get_clip_by_caption_sim,
        tags={"video", "search", "caption"},
    )


def add_mcp_resources(mcp: FastMCP):
    mcp.add_resource_fn(
        fn=list_tables,
        uri="/registry/list_tables",
        name="list_tables",
        description="List all video indexes currently available.",
        tags={"resource", "all"},
    )

    mcp.add_resource(
        name="list_table_info",
        fn=list_table_info,
        description="List information about a specific video index.",
        tags={"resource", "info"},
    )


def add_mcp_prompts(mcp: FastMCP):
    mcp.add_prompt(
        fn=struct_output_prompt,
        name="struct_output_prompt",
        description="Latest version of the response prompt from Opik.",
        tags={"prompt", "structured.output"},
    )

    mcp.add_prompt(
        fn=system_prompt,
        name="system_prompt",
        description="Latest version of the system prompt from Opik.",
        tags={"prompt", "system"},
    )


mcp = FastMCP("VideoProcessor")

add_mcp_prompts(mcp)
add_mcp_tools(mcp)
add_mcp_resources(mcp)
