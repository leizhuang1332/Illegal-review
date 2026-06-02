import pytest
import numpy as np
from uuid import uuid4
from src.illegal_review.preprocessing_layer.stages.frame_sample_stage import FrameSampleStage
from src.illegal_review.preprocessing_layer.context import PipelineContext
from src.illegal_review.preprocessing_layer.utils.frame_io import FrameStore
from src.illegal_review.config.settings import PreprocessingConfig


class TestFrameSampleStage:
    @pytest.fixture
    def config(self):
        return PreprocessingConfig(
            frame_sample_interval_short=1, frame_sample_interval_long=2,
            short_video_threshold=60, scene_change_threshold=30,
        )

    @pytest.fixture
    def ctx(self, config):
        return PipelineContext(input_id=uuid4(), video_path="/tmp/test.mp4", config=config)

    def test_name_and_dependencies(self, config):
        stage = FrameSampleStage(config)
        assert stage.name == "frame_sample"
        assert stage.dependencies == ["decode"]

    def test_short_video_interval(self, config, ctx):
        ctx.fps = 30.0; ctx.total_frames = 900
        store = FrameStore(memory_limit=1000, spill_dir="/tmp/spill")
        # Use identical frames to prevent scene-change detection from firing
        base_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        for i in range(900):
            store.append(base_frame.copy())
        ctx.raw_frames = store
        stage = FrameSampleStage(config)
        import asyncio
        asyncio.run(stage.process(ctx))
        assert ctx.sampled_frames is not None
        assert 25 <= len(ctx.sampled_frames) <= 50

    def test_long_video_interval(self, config, ctx):
        ctx.fps = 30.0; ctx.total_frames = 3600
        store = FrameStore(memory_limit=5000, spill_dir="/tmp/spill")
        # Use identical frames to prevent scene-change detection from firing
        base_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        for i in range(3600):
            store.append(base_frame.copy())
        ctx.raw_frames = store
        stage = FrameSampleStage(config)
        import asyncio
        asyncio.run(stage.process(ctx))
        assert ctx.sampled_frames is not None
        assert 50 <= len(ctx.sampled_frames) <= 100

    def test_empty_video_graceful(self, config, ctx):
        ctx.fps = 30.0; ctx.total_frames = 0
        store = FrameStore(memory_limit=100, spill_dir="/tmp/spill")
        ctx.raw_frames = store
        stage = FrameSampleStage(config)
        import asyncio
        asyncio.run(stage.process(ctx))
        assert ctx.sampled_frames is None
