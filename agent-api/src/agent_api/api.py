from fastapi import FastAPI, HTTPException
from fastmcp.client import Client
from loguru import logger

from agent_api.config import settings
from agent_api.agent import GroqAgent
from agent_api.models import (
    ChatRequest,
    ChatResponse,
    ProcessVideoRequest,
    ProcessVideoResponse,
    ResetMemoryResponse,
)

app = FastAPI(
    title="Kubrick API",
    description="An AI-powered sports assistant API using OpenAI",
    docs_url="/docs",
)


agent = GroqAgent(
    name="kubrick",
    mcp_server=settings.MCP_SERVER,
    active_tools=["process_video", "get_video_clip_from_image"],
)


@app.get("/")
async def root():
    """
    Root endpoint that redirects to API documentation
    """
    return {"message": "Welcome to Kubrick API. Visit /docs for documentation"}


@app.post("/process-video")
async def process_video(request: ProcessVideoRequest):
    """
    Process a video and return the results
    """
    mcp_client = Client(settings.MCP_SERVER)
    async with mcp_client:
        result = await mcp_client.call_tool(
            "process_video", {"video_path": request.video_path}
        )
        logger.info(result)

    return ProcessVideoResponse(message="Video processed successfully")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat with the AI assistant

    Args:
        request: ChatRequest containing the message and optional image URL

    Returns:
        ChatResponse containing the assistant's response
    """
    await agent.setup()
    
    try:
        response = await agent.chat(request.message, request.video_path)
        return ChatResponse(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/reset-memory")
async def reset_memory():
    """
    Reset the memory of the agent
    """
    agent.reset_memory()
    return ResetMemoryResponse(message="Memory reset successfully")
