"""
系统配置文件

包含所有模块的配置参数
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List


@dataclass
class InputLayerConfig:
    """输入层配置"""
    supported_formats: List[str] = field(default_factory=lambda: ["mp4", "avi", "mov", "flv"])
    max_file_size: int = 1 * 1024 * 1024 * 1024  # 1GB
    download_timeout: int = 30
    temp_dir: str = "./temp"


@dataclass
class PreprocessingConfig:
    """预处理层配置"""
    frame_sample_interval_short: int = 1  # 短视频采样间隔（秒）
    frame_sample_interval_long: int = 2   # 长视频采样间隔（秒）
    short_video_threshold: int = 60       # 短视频阈值（秒）
    scene_change_threshold: int = 30      # 场景变化检测阈值
    audio_sample_rate: int = 16000        # 音频采样率
    ocr_languages: List[str] = field(default_factory=lambda: ["ch_sim", "en"])


@dataclass
class DecodeExtractConfig:
    """解码提取层配置"""
    ffmpeg_path: Optional[str] = None
    enable_parallel: bool = True
    temp_file_cleanup: bool = True


@dataclass
class ImageRecognitionConfig:
    """图像识别引擎配置"""
    yolo_model_path: str = "yolov8n.pt"
    nsfw_model_path: str = "nsfw_model.pt"
    face_recognition_enabled: bool = True


@dataclass
class AudioAnalysisConfig:
    """音频分析引擎配置"""
    whisper_model: str = "base"
    enable_diarization: bool = False
    sensitive_detection_enabled: bool = True


@dataclass
class TextAnalysisConfig:
    """文本分析引擎配置"""
    bert_model: str = "bert-base-chinese"
    sensitive_word_list_path: str = "config/sensitive_words.txt"
    sentiment_analysis_enabled: bool = True


@dataclass
class RuleEngineConfig:
    """规则引擎配置"""
    rules_config_path: str = "config/rules.yaml"
    whitelist_priority: int = 0
    blacklist_priority: int = 1
    threshold_priority: int = 2
    composite_priority: int = 3


@dataclass
class AIEngineConfig:
    """AI引擎配置"""
    multimodal_model_path: Optional[str] = None
    confidence_threshold: float = 0.85
    review_threshold: float = 0.7
    enable_gpu: bool = True


@dataclass
class FusionDecisionConfig:
    """融合决策层配置"""
    rule_weight: float = 0.6
    ai_weight: float = 0.4
    confidence_calibration_enabled: bool = True


@dataclass
class DataStorageConfig:
    """数据存储配置"""
    postgresql_url: str = "postgresql://user:password@localhost:5432/illegal_review"
    redis_url: str = "redis://localhost:6379/0"
    object_storage_url: str = "http://localhost:9000"
    object_storage_bucket: str = "illegal-review"


@dataclass
class MessageQueueConfig:
    """消息队列配置"""
    kafka_brokers: List[str] = field(default_factory=lambda: ["localhost:9092"])
    video_tasks_topic: str = "video_tasks"
    audit_results_topic: str = "audit_results"
    feedback_events_topic: str = "feedback_events"


@dataclass
class MonitoringConfig:
    """监控配置"""
    prometheus_port: int = 9090
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class SystemConfig:
    """系统主配置"""
    input_layer: InputLayerConfig = field(default_factory=InputLayerConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    decode_extract: DecodeExtractConfig = field(default_factory=DecodeExtractConfig)
    image_recognition: ImageRecognitionConfig = field(default_factory=ImageRecognitionConfig)
    audio_analysis: AudioAnalysisConfig = field(default_factory=AudioAnalysisConfig)
    text_analysis: TextAnalysisConfig = field(default_factory=TextAnalysisConfig)
    rule_engine: RuleEngineConfig = field(default_factory=RuleEngineConfig)
    ai_engine: AIEngineConfig = field(default_factory=AIEngineConfig)
    fusion_decision: FusionDecisionConfig = field(default_factory=FusionDecisionConfig)
    data_storage: DataStorageConfig = field(default_factory=DataStorageConfig)
    message_queue: MessageQueueConfig = field(default_factory=MessageQueueConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)


# 全局配置实例
config = SystemConfig()
