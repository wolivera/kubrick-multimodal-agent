from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    video_path: str


class ChatResponse(BaseModel):
    response: str


class ProcessVideoRequest(BaseModel):
    video_path: str


class ProcessVideoResponse(BaseModel):
    message: str
    task_id: str


class ResetMemoryResponse(BaseModel):
    message: str
