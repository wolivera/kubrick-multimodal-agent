from pydantic import BaseModel, Field

class RoutingResponseModel(BaseModel):
    tool_use: bool = Field(description="Whether the user's question requires a tool call.")
