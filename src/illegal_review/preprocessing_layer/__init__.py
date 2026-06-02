"""
预处理层模块

DAG Pipeline 架构：
1. 视频解码 (DecodeStage)：FFmpeg pipe 模式解码为帧序列
2. 音频提取 (AudioExtractStage)：提取音频 → 16kHz 单声道 PCM
3. 帧采样 (FrameSampleStage)：自适应间隔 + 场景变化检测
4. 语音转写 (SpeechStage)：Whisper small 模型
5. 文字识别 (OCRStage)：EasyOCR 中英文
"""

from src.illegal_review.preprocessing_layer.service import PreprocessingService

__all__ = ["PreprocessingService"]
