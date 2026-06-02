"""
解码提取层模块

负责将视频文件解码为可分析的基础单元，提取帧序列、音频轨道和文本信息。

核心模块：
- VideoDecoder：视频解码、帧序列生成
- AudioExtractor：音频轨道提取、格式转换
- FrameSampler：帧采样策略执行、场景变化检测
- ContentRecognizer：语音转写、OCR文字识别
"""
