"""
数据模型定义

定义系统中使用的核心数据结构
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import UUID


# 输入层数据模型
class VideoMetadata(BaseModel):
    """视频元数据"""
    duration: float = Field(description="视频时长（秒）")
    fps: float = Field(description="帧率")
    width: int = Field(description="宽度（像素）")
    height: int = Field(description="高度（像素）")
    codec: str = Field(description="视频编码格式")
    audio_codec: Optional[str] = Field(description="音频编码格式")
    bitrate: Optional[int] = Field(description="比特率（bps）")


class SourceInfo(BaseModel):
    """源信息"""
    original_source: str = Field(description="原始输入（文件路径/URL/流地址）")
    file_size: Optional[int] = Field(description="文件大小（字节）")
    content_type: Optional[str] = Field(description="MIME类型")


class SegmentInfo(BaseModel):
    """片段信息"""
    start_time: float = Field(description="起始时间（秒）")
    end_time: float = Field(description="结束时间（秒）")
    duration: float = Field(description="片段时长（秒）")


class InputResult(BaseModel):
    """输入层输出结果"""
    input_id: UUID = Field(description="唯一标识")
    input_type: str = Field(description="输入类型：file/url/stream/segment")
    source_info: SourceInfo = Field(description="源信息")
    video_metadata: VideoMetadata = Field(description="视频元数据")
    temp_path: str = Field(description="临时文件路径")
    segment_info: Optional[SegmentInfo] = Field(description="片段信息")
    status: str = Field(description="状态：success/failed")
    error_message: Optional[str] = Field(description="错误信息")
    created_at: datetime = Field(description="创建时间")


# 预处理层数据模型
class FrameData(BaseModel):
    """帧数据"""
    frame_index: int = Field(description="帧索引")
    timestamp: float = Field(description="时间戳（秒）")
    image_data: bytes = Field(description="图像数据")
    width: int = Field(description="宽度")
    height: int = Field(description="高度")


class AudioData(BaseModel):
    """音频数据"""
    audio_path: str = Field(description="音频文件路径")
    sample_rate: int = Field(description="采样率（Hz）")
    duration: float = Field(description="音频时长（秒）")
    channels: int = Field(description="声道数")


class TranscriptSegment(BaseModel):
    """转录片段"""
    text: str = Field(description="文本内容")
    start: float = Field(description="开始时间（秒）")
    end: float = Field(description="结束时间（秒）")
    speaker: Optional[str] = Field(description="说话人标识")


class OCRResult(BaseModel):
    """OCR识别结果"""
    text: str = Field(description="识别文字")
    confidence: float = Field(description="置信度")
    bbox: Optional[List[int]] = Field(description="边界框")
    frame_index: int = Field(description="所在帧索引")


class PreprocessingResult(BaseModel):
    """预处理层输出结果"""
    input_id: UUID = Field(description="追踪ID")
    frames: List[FrameData] = Field(description="采样后的帧序列")
    audio: AudioData = Field(description="音频数据")
    transcript: str = Field(description="完整转录文本")
    transcript_segments: List[TranscriptSegment] = Field(description="分段转录结果")
    ocr_results: List[OCRResult] = Field(description="OCR识别结果")
    metadata: VideoMetadata = Field(description="处理元数据")
    processing_stats: Dict[str, float] = Field(description="处理统计信息")


# 分析引擎层数据模型
class ViolationDetection(BaseModel):
    """违规检测结果"""
    type: str = Field(description="违规类型")
    category: str = Field(description="违规类别")
    confidence: float = Field(description="检测置信度")
    timestamp: Optional[float] = Field(description="违规时间戳")
    context: Optional[str] = Field(description="上下文信息")
    evidence: Optional[Dict[str, Any]] = Field(description="证据详情")


class ImageAnalysisResult(BaseModel):
    """图像分析结果"""
    video_id: UUID = Field(description="视频标识")
    violations: List[ViolationDetection] = Field(description="违规检测列表")
    features: Dict[str, Any] = Field(description="视觉特征")


class AudioAnalysisResult(BaseModel):
    """音频分析结果"""
    video_id: UUID = Field(description="视频标识")
    transcript: str = Field(description="转写文本")
    violations: List[ViolationDetection] = Field(description="违规检测列表")
    features: Dict[str, Any] = Field(description="音频特征")


class TextAnalysisResult(BaseModel):
    """文本分析结果"""
    video_id: UUID = Field(description="视频标识")
    semantic_embedding: Optional[List[float]] = Field(description="语义嵌入")
    sensitive_words: List[str] = Field(description="敏感词列表")
    sentiment_score: Optional[float] = Field(description="情感分数")
    entities: List[Dict[str, Any]] = Field(description="实体列表")


class RuleEngineResult(BaseModel):
    """规则引擎结果"""
    video_id: UUID = Field(description="视频标识")
    decision: str = Field(description="决策：pass/reject/pending")
    rule_id: Optional[str] = Field(description="匹配的规则ID")
    rule_name: Optional[str] = Field(description="规则名称")
    evidence: Optional[Dict[str, Any]] = Field(description="证据")
    confidence: float = Field(description="置信度")


class AIEngineResult(BaseModel):
    """AI引擎结果"""
    video_id: UUID = Field(description="视频标识")
    probabilities: Dict[str, float] = Field(description="各类别概率")
    features: Dict[str, Any] = Field(description="多模态特征")


# 融合决策层数据模型
class DecisionEvidence(BaseModel):
    """决策证据"""
    type: str = Field(description="证据类型")
    source: str = Field(description="证据来源")
    confidence: float = Field(description="置信度")
    description: str = Field(description="证据描述")
    timestamp: Optional[float] = Field(description="时间戳")
    bbox: Optional[List[int]] = Field(description="边界框")


class FinalDecision(BaseModel):
    """最终决策结果"""
    video_id: UUID = Field(description="视频唯一标识")
    decision: str = Field(description="审核结果：pass/reject/review")
    violation_type: str = Field(description="违规类型")
    confidence: float = Field(description="置信度分数")
    confidence_level: str = Field(description="置信度等级：high/medium/low")
    evidence: Dict[str, Any] = Field(description="违规证据")
    evidence_items: List[DecisionEvidence] = Field(description="证据项列表")
    suggestion: str = Field(description="建议操作")
    processing_time: float = Field(description="处理耗时（毫秒）")
    sources: List[str] = Field(description="决策来源")
    timestamp: datetime = Field(description="处理时间戳")
    request_id: UUID = Field(description="请求唯一标识")


# 反馈闭环数据模型
class FeedbackData(BaseModel):
    """反馈数据"""
    video_id: UUID = Field(description="视频ID")
    original_decision: str = Field(description="原始决策")
    corrected_decision: str = Field(description="修正决策")
    correction_reason: Optional[str] = Field(description="修正原因")
    corrector_id: str = Field(description="审核人员ID")
    correction_time: datetime = Field(description="修正时间")
    confidence: float = Field(description="原始置信度")
    violation_type: str = Field(description="违规类型")
