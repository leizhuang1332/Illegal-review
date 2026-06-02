import pytest
import asyncio
from uuid import uuid4
from src.illegal_review.preprocessing_layer.pipeline import Pipeline
from src.illegal_review.preprocessing_layer.stage import Stage
from src.illegal_review.preprocessing_layer.context import PipelineContext
from src.illegal_review.preprocessing_layer.exceptions import PipelineError
from src.illegal_review.config.settings import PreprocessingConfig


class _PassStage(Stage):
    def __init__(self, name, deps=None):
        self._name = name
        self._deps = deps or []
    @property
    def name(self) -> str:
        return self._name
    @property
    def dependencies(self) -> list[str]:
        return self._deps
    async def process(self, ctx):
        ctx.stats[f"{self._name}_done"] = 1.0


class _FailStage(Stage):
    def __init__(self, name, deps=None):
        self._name = name
        self._deps = deps or []
    @property
    def name(self) -> str:
        return self._name
    @property
    def dependencies(self) -> list[str]:
        return self._deps
    async def process(self, ctx):
        raise RuntimeError(f"{self._name} failed")


class _RecordStage(Stage):
    _exec_order = []
    def __init__(self, name, deps=None):
        self._name = name
        self._deps = deps or []
    @property
    def name(self) -> str:
        return self._name
    @property
    def dependencies(self) -> list[str]:
        return self._deps
    async def process(self, ctx):
        _RecordStage._exec_order.append(self._name)
        await asyncio.sleep(0.01)


class TestPipeline:
    @pytest.fixture
    def ctx(self):
        return PipelineContext(
            input_id=uuid4(), video_path="/tmp/test.mp4",
            config=PreprocessingConfig(),
        )

    def test_single_stage(self, ctx):
        pipeline = Pipeline()
        pipeline.add_stage(_PassStage("only_stage"))
        asyncio.run(pipeline.run(ctx))
        assert ctx.stats["only_stage_done"] == 1.0

    def test_linear_chain(self, ctx):
        pipeline = Pipeline()
        pipeline.add_stage(_PassStage("a"))
        pipeline.add_stage(_PassStage("b", deps=["a"]))
        pipeline.add_stage(_PassStage("c", deps=["b"]))
        asyncio.run(pipeline.run(ctx))
        assert ctx.stats["a_done"] == 1.0
        assert ctx.stats["b_done"] == 1.0
        assert ctx.stats["c_done"] == 1.0

    def test_parallel_branches(self, ctx):
        _RecordStage._exec_order.clear()
        pipeline = Pipeline()
        pipeline.add_stage(_RecordStage("root"))
        pipeline.add_stage(_RecordStage("left", deps=["root"]))
        pipeline.add_stage(_RecordStage("right", deps=["root"]))
        asyncio.run(pipeline.run(ctx))
        exec_order = _RecordStage._exec_order
        assert exec_order[0] == "root"
        assert "left" in exec_order[1:] and "right" in exec_order[1:]

    def test_upstream_failure_skips_downstream(self, ctx):
        pipeline = Pipeline()
        pipeline.add_stage(_FailStage("root"))
        pipeline.add_stage(_PassStage("child", deps=["root"]))
        asyncio.run(pipeline.run(ctx))
        assert "child_done" not in ctx.stats
        assert len(ctx.errors) > 0

    def test_branch_isolation(self, ctx):
        pipeline = Pipeline()
        pipeline.add_stage(_PassStage("root"))
        pipeline.add_stage(_FailStage("bad_branch", deps=["root"]))
        pipeline.add_stage(_PassStage("good_branch", deps=["root"]))
        asyncio.run(pipeline.run(ctx))
        assert ctx.stats["good_branch_done"] == 1.0
        assert "bad_branch_done" not in ctx.stats

    def test_circular_dependency_detected(self, ctx):
        pipeline = Pipeline()
        pipeline.add_stage(_PassStage("a", deps=["b"]))
        pipeline.add_stage(_PassStage("b", deps=["a"]))
        with pytest.raises(PipelineError, match="Circular"):
            asyncio.run(pipeline.run(ctx))

    def test_duplicate_name_raises(self):
        pipeline = Pipeline()
        pipeline.add_stage(_PassStage("dup"))
        with pytest.raises(ValueError, match="already registered"):
            pipeline.add_stage(_PassStage("dup"))

    def test_stage_timing(self, ctx):
        pipeline = Pipeline()
        pipeline.add_stage(_PassStage("timed"))
        asyncio.run(pipeline.run(ctx))
        assert "timed_duration_ms" in ctx.stats
        assert ctx.stats["timed_duration_ms"] >= 0
