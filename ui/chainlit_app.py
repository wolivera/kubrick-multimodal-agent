import os
import shutil

import aiohttp
import chainlit as cl

API_BASE_URL = "http://agent-api:8080"  # Adjust this based on your API location

os.makedirs("videos", exist_ok=True)


@cl.on_chat_start
async def start():
    files = None

    while not files:
        files = await cl.AskFileMessage(
            content="Please upload your video to begin!",
            accept=[".mp4"],
            max_size_mb=100,
        ).send()

    video_file = files[0]

    try:
        dest_path = os.path.join("videos", video_file.name)

        shutil.move(video_file.path, dest_path)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{API_BASE_URL}/process-video",
                json={"video_path": dest_path},
            ) as response:
                if response.status == 200:
                    elements = [
                        cl.Video(name=video_file.name, path=dest_path, display="inline"),
                    ]
                    await cl.Message(
                        content="You video has been correctly uploaded!!",
                        elements=elements,
                    ).send()
                else:
                    error_text = await response.text()
                    await cl.Message(content=f"Error from API: {error_text}").send()
        cl.user_session.set("video_path", dest_path)

    except Exception as e:
        await cl.Message(content=f"Error handling video file: {str(e)}").send()


@cl.on_message
async def main(message: cl.Message):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{API_BASE_URL}/chat",
                json={"message": message.content, "video_path": cl.user_session.get("video_path")},
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    await cl.Message(content=data["response"]).send()
                else:
                    error_text = await response.text()
                    await cl.Message(content=f"Error from API: {error_text}").send()
    except Exception as e:
        await cl.Message(content=f"Error communicating with API: {str(e)}").send()
