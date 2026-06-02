class PreprocessingError(Exception):
    """预处理层基础异常"""
    pass


class DecodeError(PreprocessingError):
    """视频解码失败"""
    pass


class AudioExtractError(PreprocessingError):
    """音频提取失败"""
    pass


class RecognitionError(PreprocessingError):
    """内容识别失败（语音/OCR）"""
    pass


class PipelineError(PreprocessingError):
    """Pipeline 调度错误（如循环依赖、配置错误）"""
    pass
