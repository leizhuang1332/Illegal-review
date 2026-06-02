from abc import ABC, abstractmethod
from src.illegal_review.preprocessing_layer.context import PipelineContext


class Stage(ABC):
    """所有处理阶段的抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """阶段名称，用作 DAG 节点标识"""
        ...

    @property
    @abstractmethod
    def dependencies(self) -> list[str]:
        """依赖的阶段名称列表"""
        ...

    @abstractmethod
    async def process(self, ctx: PipelineContext) -> None:
        """执行阶段处理逻辑，结果写入 ctx"""
        ...
