import glob
import os
from pathlib import Path
from typing import List, Tuple

import pixeltable as pxt
from core.functions import caption_video, compose_semantics
from core.splitter import split_video_to_chunks_subprocess
from pixeltable.functions import whisper
from pixeltable.functions.audio import get_metadata
from pixeltable.functions.huggingface import sentence_transformer
from pixeltable.functions.video import extract_audio
from pixeltable.iterators.string import StringSplitter
from pixeltable.iterators.video import FrameIterator

DEFAULT_TMP_PATH = Path(os.getcwd()) / "/.cache/tmp"


class VideoProcessor:
    def __init__(self, pxt_cache: str = "poc", videos_cache: Path = DEFAULT_TMP_PATH):
        self.pxt_cache = pxt_cache
        self.videos_cache = videos_cache

    def preprocess_video(self, video_path, chunk_duration: int) -> Tuple[str, List[str]]:
        vpath = Path(video_path)
        assert vpath.exists(), f"Video could not be found {video_path}"

        if Path(self.videos_cache).exists():
            clips_cache = self.videos_cache
        clips_cache = split_video_to_chunks_subprocess(
            video_path=vpath, chunk_duration=chunk_duration, cache_path=self.videos_cache
        )
        video_files = glob.glob(str(clips_cache) + "/*.mp4")
        video_files.sort()

        return clips_cache, video_files

    def add_video(self, video_path: str):
        self.video_table.insert([{"video": video_path}])

    def compose_pixeltable(self):
        """Sets up the PixelTable environment and data."""
        pxt.drop_dir(self.pxt_cache, force=True)
        pxt.create_dir(self.pxt_cache, if_exists="ignore")

        self.video_table = pxt.create_table(
            f"{self.pxt_cache}.data",
            {"video": pxt.Video},
            if_exists="ignore",
        )

        # STEP 1 : get audioe from video
        self.video_table.add_computed_column(
            audio=extract_audio(self.video_table.video, format="mp3"),
            if_exists="ignore",
        )

        # STEP 2: transcribe audio channel
        self.video_table.add_computed_column(
            transcription=whisper.transcribe(audio=self.video_table.audio, model="base.en"),
            if_exists="ignore",
        )

        # Step 3: get metadata from audio (gibberish but useful for debug/view)
        self.video_table.add_computed_column(metadata=get_metadata(self.video_table.audio), if_exists="ignore")

        # Step 4: caption an entire video
        self.video_table.add_computed_column(
            video_caption=caption_video(video=self.video_table.video, prompt="What's happening in the video?")
        )

        # Step 5: Combine semantics (caption + transcription)
        self.video_table.add_computed_column(
            fulltext=compose_semantics(self.video_table.video_caption, self.video_table.transcription),
            if_exists="replace",
        )

        # Step 6: Split the text into sentences
        self.sentences_view = pxt.create_view(
            f"{self.pxt_cache}.sentences",
            self.video_table,
            iterator=StringSplitter.create(text=self.video_table.fulltext, separators="sentence"),
            if_exists="ignore",
        )

        # Step 7: Add the embedding index for the sentences
        self.video_table.add_embedding_index(
            column="fulltext",  # this comes from L80, as iterator creates `text` rows`
            string_embed=sentence_transformer.using(model_id="intfloat/e5-large-v2"),
            if_exists="ignore",
        )

        self.frames_view = pxt.create_view(
            "frames",
            self.video_table,
            iterator=FrameIterator.create(video=self.video_table.video, fps=1.0),  # FIXME: move to config
            if_exists="ignore",
        )

    @property
    def sentences(self) -> List[str]:
        """Returns the sentences from the video."""
        return self.sentences_view.select(self.sentences_view.text).collect()

    @property
    def frames(self) -> List[str]:
        """Returns the frames from the video."""
        return self.frames_view.select(self.frames_view.frame).collect()

    @property
    def video_caption(self) -> List[str]:
        """Returns the captions from the video."""
        return self.frames_view.imcaption
