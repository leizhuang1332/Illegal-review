import pytest
from uuid import uuid4
from src.illegal_review.preprocessing_layer.stage import Stage
from src.illegal_review.preprocessing_layer.context import PipelineContext
from src.illegal_review.config.settings import PreprocessingConfig


class _ConcreteStage(Stage):
    """测试用具体实现"""
    @property
    def name(self) -> str:
        return "test_stage"

    @property
    def dependencies(self) -> list[str]:
        return ["upstream_stage"]

    async def process(self, ctx: PipelineContext) -> None:
        ctx.stats["test_duration_ms"] = 42.0


class TestStageAbstract:
    @pytest.fixture
    def ctx(self):
        return PipelineContext(
            input_id=uuid4(),
            video_path="/tmp/test.mp4",
            config=PreprocessingConfig(),
        )

    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            Stage()

    def test_concrete_stage_has_name(self):
        stage = _ConcreteStage()
        assert stage.name == "test_stage"

    def test_concrete_stage_has_dependencies(self):
        stage = _ConcreteStage()
        assert stage.dependencies == ["upstream_stage"]

    @pytest.mark.asyncio
    async def test_concrete_stage_process(self, ctx):
        stage = _ConcreteStage()
        await stage.process(ctx)
        assert ctx.stats["test_duration_ms"] == 42.0

    def test_missing_name_raises(self):
        class BadStage(Stage):
            dependencies = []
            async def process(self, ctx): pass
        with pytest.raises(TypeError):
            BadStage()

    def test_missing_dependencies_raises(self):
        class BadStage(Stage):
            name = "bad"
            async def process(self, ctx): pass
        with pytest.raises(TypeError):
            BadStage()
