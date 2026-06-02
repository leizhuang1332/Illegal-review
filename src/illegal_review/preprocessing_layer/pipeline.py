import asyncio
import logging
import time
from src.illegal_review.preprocessing_layer.stage import Stage
from src.illegal_review.preprocessing_layer.context import PipelineContext
from src.illegal_review.preprocessing_layer.exceptions import PipelineError, PreprocessingError

logger = logging.getLogger(__name__)


class Pipeline:
    """DAG Pipeline 调度器 — 拓扑并行执行 + 失败隔离"""

    def __init__(self):
        self._stages: dict[str, Stage] = {}

    def add_stage(self, stage: Stage) -> None:
        if stage.name in self._stages:
            raise ValueError(f"Stage '{stage.name}' already registered")
        self._stages[stage.name] = stage

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        t0 = time.perf_counter()
        completed: set[str] = set()
        failed: set[str] = set()
        all_names = set(self._stages.keys())

        for stage in self._stages.values():
            for dep in stage.dependencies:
                if dep not in all_names:
                    raise PipelineError(
                        f"Stage '{stage.name}' depends on '{dep}' which is not registered"
                    )

        while len(completed) + len(failed) < len(self._stages):
            ready = {
                name for name, stage in self._stages.items()
                if name not in completed
                and name not in failed
                and all(d in completed for d in stage.dependencies)
                and not any(d in failed for d in stage.dependencies)
            }

            if not ready:
                remaining = all_names - completed - failed
                # 阶段因依赖失败而被阻塞 → 静默跳过
                if all(
                    any(d in failed for d in self._stages[name].dependencies)
                    for name in remaining
                ):
                    for name in remaining:
                        failed.add(name)
                        logger.info(f"Stage '{name}' skipped — upstream failure")
                    break

                raise PipelineError(
                    f"Circular dependency or deadlock detected. "
                    f"Remaining stages: {remaining}"
                )

            ready_stages = [(name, self._stages[name]) for name in ready]
            results = await asyncio.gather(
                *[self._run_stage(name, stage, ctx) for name, stage in ready_stages],
                return_exceptions=True,
            )

            for (name, _), result in zip(ready_stages, results):
                if isinstance(result, Exception):
                    failed.add(name)
                    ctx.errors.append(f"{name}: {result}")
                    logger.error(f"Stage '{name}' failed: {result}")
                else:
                    completed.add(name)

        ctx.stats["total_duration_ms"] = (time.perf_counter() - t0) * 1000

        if "decode" in failed:
            raise PreprocessingError(
                f"Decode stage failed, pipeline aborted. Errors: {ctx.errors}"
            )

        return ctx

    async def _run_stage(self, name: str, stage: Stage, ctx: PipelineContext) -> None:
        t0 = time.perf_counter()
        await stage.process(ctx)
        elapsed = (time.perf_counter() - t0) * 1000
        ctx.stats[f"{name}_duration_ms"] = elapsed
        logger.info(f"Stage '{name}' completed in {elapsed:.0f}ms")
