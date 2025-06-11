import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pixeltable as pxt
from loguru import logger
from pixeltable.functions import whisper
from pixeltable.functions.huggingface import clip, sentence_transformer
from pixeltable.functions.video import extract_audio
from pixeltable.iterators import AudioSplitter
from pixeltable.iterators.video import FrameIterator

import mcp_server.video.ingestion.registry as registry
from mcp_server.config import get_settings
from mcp_server.video.ingestion.functions import caption_image, extract_text_from_chunk

if TYPE_CHECKING:
    from mcp_server.video.ingestion.models import CachedTable

logger = logger.bind(name="VideoProcessor")
settings = get_settings()


class VideoProcessor:
    def __init__(
        self,
    ):
        self._pxt_cache: Optional[str] = None
        self._video_table = None
        self._frames_view = None
        self._audio_chunks = None
        self._video_mapping_idx: Optional[str] = None

        logger.info(
            "VideoProcessor initialized",
            f"\n Split FPS: {settings.SPLIT_FPS}",
            f"\n Audio Chunk: {settings.AUDIO_CHUNK_LENGTH} seconds",
        )

    def setup_table(self, video_name: str):
        self._video_mapping_idx = video_name
        exists = self._check_if_exists()
        if exists:
            logger.info(f"Video index '{self._video_mapping_idx}' already exists and is ready for use.")
            cached_table: "CachedTable" = registry.get_table(self._video_mapping_idx)
            self.pxt_cache = cached_table.video_cache
            self.video_table = cached_table.video_table
            self.frames_view = cached_table.frames_view
            self.audio_chunks = cached_table.audio_chunks_view

        else:
            self.pxt_cache = f"cache_{uuid.uuid4().hex[-4:]}"
            self.video_table_name = f"{self.pxt_cache}.table"
            self.frames_view_name = f"{self.video_table_name}_frames"
            self.audio_view_name = f"{self.video_table_name}_audio_chunks"
            self.video_table = None

            self._setup_table()

            registry.add_index_to_registry(
                video_name=self._video_mapping_idx,
                video_cache=self.pxt_cache,
                frames_view_name=self.frames_view_name,
                audio_view_name=self.audio_view_name,
            )
            logger.info(f"Creating new video index '{self.video_table_name}' in '{self.pxt_cache}'")

    def _check_if_exists(self) -> bool:
        """
        Checks if the PixelTable table and related views/index for the video index exist.

        Returns:
            bool: True if all components exist, False otherwise.
        """
        existing_tables = registry.get_registry()
        return self._video_mapping_idx in existing_tables

    def _setup_table(self):
        logger.info(f"Creating cache path {self.pxt_cache}.")
        Path(self.pxt_cache).mkdir(parents=True, exist_ok=True)
        pxt.create_dir(self.pxt_cache, if_exists="replace_force")

        self.video_table = pxt.create_table(
            self.video_table_name,
            schema={"video": pxt.Video},
            if_exists="replace_force",
        )

        self.video_table.add_computed_column(
            audio_extract=extract_audio(self.video_table.video, format="mp3"),
            if_exists="ignore",
        )

        self.audio_chunks = pxt.create_view(
            self.audio_view_name,
            self.video_table,
            iterator=AudioSplitter.create(
                audio=self.video_table.audio_extract,
                chunk_duration_sec=settings.AUDIO_CHUNK_LENGTH,
                overlap_sec=settings.AUDIO_OVERLAP_SECONDS,
                min_chunk_duration_sec=settings.AUDIO_CHUNK_LENGTH - settings.AUDIO_OVERLAP_SECONDS,
            ),
            if_exists="ignore",
        )

        self.audio_chunks.add_computed_column(
            transcription=whisper.transcribe(audio=self.audio_chunks.audio_chunk, model="base.en"),
            if_exists="ignore",
        )
        self.audio_chunks.add_computed_column(
            chunk_text=extract_text_from_chunk(self.audio_chunks.transcription),
            if_exists="ignore",
        )

        self.audio_chunks.add_embedding_index(
            column=self.audio_chunks.chunk_text,
            string_embed=sentence_transformer.using(model_id=settings.TRANSCRIPT_SIMILARITY_EMBD_MODEL),
            if_exists="ignore",
            idx_name="chunks_index",
        )

        self.frames_view = pxt.create_view(
            self.frames_view_name,
            self.video_table,
            iterator=FrameIterator.create(video=self.video_table.video, fps=settings.SPLIT_FPS),
            if_exists="ignore",
        )

        # Add semantic index for raw frames and over their captions using the same CLIP model
        # CLIP is contrastive, embeddings of text-image pairs are close in the embedding space
        self.frames_view.add_embedding_index(
            column=self.frames_view.frame,
            image_embed=clip.using(model_id=settings.IMAGE_SIMILARITY_EMBD_MODEL),
        )

        self.frames_view.add_computed_column(
            im_caption=caption_image(
                image=self.frames_view.frame,
                prompt=settings.CAPTION_MODEL_PROMPT,
            ),
            if_exists="ignore",
        )

        self.frames_view.add_embedding_index(
            column=self.frames_view.im_caption,
            string_embed=clip.using(model_id=settings.CAPTION_SIMILARITY_EMBD_MODEL),
        )

    def add_video(self, video_path: str) -> bool:
        """
        Add a video to the pixel table.

        Args:
            video_path (str): The path to the video file.
        """
        if not self.video_table:
            raise ValueError("Video table is not initialized. Call setup_table() first.")

        logger.info(f"Adding video {video_path} to table {self.video_table_name}")
        self.video_table.insert([{"video": video_path}])

        return True
