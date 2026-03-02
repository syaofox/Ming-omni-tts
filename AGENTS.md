# AGENTS.md - Ming-omni-tts 开发指南

## 项目概述

Ming-omni-tts 是一个高性能的统一音频生成模型，支持语音、音乐和声音的生成与控制。基于 PyTorch 2.6.0 和 CUDA 12.4 开发。

## 环境设置

### Docker 环境（推荐）

```bash
# 构建 Docker 镜像
docker compose build

# 进入容器交互
docker compose run --rm ming-omni-tts
```

### 宿主机要求

- NVIDIA GPU (RTX 3060+ 推荐)
- Docker + nvidia-docker 或 NVIDIA Container Toolkit
- 模型文件放在 `./models/Ming-omni-tts-0.5B` 目录

## 构建与运行命令

### 运行测试

```bash
# 运行完整测试（所有示例）
docker compose run --rm ming-omni-tts python cookbooks/test.py

# 运行单个测试用例 - TTS 测试
docker compose run --rm ming-omni-tts python -c "
import sys; sys.path.insert(0, '.')
from webui import MingAudio
model = MingAudio('./models/Ming-omni-tts-0.5B')
model.speech_generation(
    prompt='Please generate speech based on the following description.\n',
    text='你好世界',
    use_zero_spk_emb=True,
    output_wav_path='output/test.wav'
)"

# 运行单个测试用例 - TTA 测试
docker compose run --rm ming-omni-tts python -c "
import sys; sys.path.insert(0, '.')
from webui import MingAudio
model = MingAudio('./models/Ming-omni-tts-0.5B')
model.speech_generation(
    prompt='Please generate audio events based on given text.\n',
    text='Thunder and a gentle rain',
    max_decode_steps=200, cfg=4.5, sigma=0.3, temperature=2.5,
    output_wav_path='output/tta.wav'
)"

# 运行单个测试用例 - BGM 测试
docker compose run --rm ming-omni-tts python -c "
import sys; sys.path.insert(0, '.')
from webui import MingAudio
model = MingAudio('./models/Ming-omni-tts-0.5B')
model.speech_generation(
    prompt='Please generate music based on the following description.\n',
    text='Genre: 电子舞曲 Mood: 欢快 Instrument: 架子鼓',
    max_decode_steps=400,
    output_wav_path='output/bgm.wav'
)"
```

### WebUI 启动

```bash
# 启动 WebUI（仅 Gradio）
docker compose run --rm -p 7860:7860 ming-omni-tts python webui.py

# 启动 API（仅 Flask）
docker compose run --rm -p 7860:7860 ming-omni-tts python api.py

# 启动 WebUI（无模型测试模式）
docker compose run --rm -p 7860:7860 -e LOAD_MODEL=false ming-omni-tts python webui.py
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| MODEL_PATH | ./models/Ming-omni-tts-0.5B | 模型路径 |
| PORT | 7860 | 服务端口 |
| LOAD_MODEL | true | 是否加载模型 |

## Lint 命令

```bash
# Python 语法检查
docker compose run --rm ming-omni-tts python -m py_compile <file.py>

# ruff 检查（安装后）
docker compose run --rm ming-omni-tts pip install ruff && ruff check .

# black 格式化检查（安装后）
docker compose run --rm ming-omni-tts pip install black && black --check .

# 语法检查多个文件
docker compose run --rm ming-omni-tts python -m py_compile webui.py api.py
```

## 代码风格指南

### 文件头

所有 Python 文件必须包含以下文件头：

```python
#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.
```

### 导入顺序

按以下顺序组织导入：

1. Python 标准库 (os, sys, json, typing, etc.)
2. 第三方库 (torch, transformers, gradio, flask, loguru)
3. 本地模块 (from .xxx import, from cookbooks.xxx)

```python
import os
import sys
import json
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import gradio as gr
from flask import Flask, request
from loguru import logger
from transformers import PreTrainedModel

from modeling_bailingmm import BailingMMNativeForConditionalGeneration
from sentence_manager.sentence_manager import SentenceNormalizer
```

### 命名规范

- **类名**: PascalCase (如 `MingAudio`, `SpkembExtractor`, `BailingMMNativeForConditionalGeneration`)
- **函数/变量**: snake_case (如 `seed_everything`, `model_path`, `speech_generation`)
- **常量**: 全大写 snake_case (如 `MAX_DECODE_STEPS`, `OUTPUT_DIR`)
- **私有方法/变量**: 前缀下划线 (如 `_extract_spk_embedding`, `_init_model`)

### 类型注解

**必须**为新代码添加类型注解：

```python
def generate_speech(
    text: str,
    prompt_wav: Optional[str] = None,
    max_decode_steps: int = 200
) -> torch.Tensor:
    ...

class MingAudio:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.device: str = device
        self.model: Optional[torch.nn.Module] = None
        self.sample_rate: int = 0
```

### 错误处理

使用 loguru 进行日志记录，不要使用 print：

```python
from loguru import logger

try:
    result = model.generate(...)
except Exception as e:
    logger.error(f"生成失败: {e}")
    raise

logger.warning("GPU 内存不足，切换到 CPU 模式")
logger.info(f"Model loaded from {model_path}")
```

### 代码格式化

- 缩进: 4 空格
- 行长度: 推荐 100 字符以内，最大不超过 120 字符
- 字符串引号: 双引号优先
- 类之间空 2 行，函数之间空 1 行

### 文档字符串

为所有公共方法和类添加文档字符串：

```python
class MingAudio:
    """Ming-Omni-TTS 语音合成模型封装类"""
    
    def speech_generation(
        self,
        prompt: str,
        text: str,
        use_spk_emb: bool = False,
        use_zero_spk_emb: bool = False,
        instruction: Optional[dict] = None,
        prompt_wav_path: Optional[str] = None,
        max_decode_steps: int = 200,
        cfg: float = 2.0,
        sigma: float = 0.25,
        temperature: float = 0.0,
        output_wav_path: Optional[str] = None,
    ) -> torch.Tensor:
        """
        生成语音音频
        
        Args:
            prompt: 提示词
            text: 输入文本
            use_spk_emb: 是否使用说话人embedding
            use_zero_spk_emb: 是否使用零说话人embedding
            instruction: 声音控制指令
            prompt_wav_path: 参考音频路径
            max_decode_steps: 最大解码步数
            cfg: CFG 强度
            sigma: 方差参数
            temperature: 温度参数
            output_wav_path: 输出音频路径
        
        Returns:
            torch.Tensor: 生成的波形
        """
```

## 项目结构

```
Ming-omni-tts/
├── modeling_bailingmm.py      # 主模型定义
├── modeling_bailing_moe.py    # MoE 模型
├── configuration_bailingmm.py # 配置类
├── webui.py                   # WebUI 界面 + MingAudio 类
├── api.py                     # API 服务（可独立启动）
├── spkemb_extractor.py        # 说话人embedding提取
├── audio_tokenizer/           # 音频 tokenizer
│   ├── modeling_audio_vae.py
│   └── audio_encoder.py
├── fm/                        # 流匹配模块
│   ├── dit.py
│   ├── flowloss.py
│   └── CFM.py
├── sentence_manager/          # 文本规范化
├── cookbooks/
│   ├── test.py               # 测试脚本
│   └── instructions.md       # 指令说明
└── models/                    # 模型文件目录
```

## 推理类别

| 类别 | Prompt | 说明 |
|------|--------|------|
| 语音合成 (TTS) | "Please generate speech based on the following description.\n" | 文本转语音 |
| 声音事件 (TTA) | "Please generate audio events based on given text.\n" | 声音事件合成 |
| 背景音乐 (BGM) | "Please generate music based on the following description.\n" | 音乐生成 |

## 声音控制指令格式

```python
instruction = {
    "audio_sequence": [{
        "序号": 1,
        "说话人": "speaker_1",
        "音色描述": "一位温柔的母亲声音，音色低沉浑厚",
        "情感": "高兴",      # 高兴/悲伤/愤怒/惊讶/恐惧/厌恶
        "方言": "普通话",    # 普通话/广粤话/四川话/东北话/河南话
        "风格": "新闻播报",  # 正式/casual/新闻播报/讲故事/ASMR耳语
        "语速": 1.0,        # 0.5-2.0
        "基频": 1.0,        # 音高 0.5-2.0
        "音量": 1.0,        # 0.5-2.0
        "IP": "灵小甄",     # 内置音色（可选）
        "BGM": {           # 背景音乐/音效
            "Genre": "电子舞曲",
            "Mood": "欢快",
            "Instrument": "架子鼓",
            "Theme": "节日",
            "ENV": "鸟鸣",
            "SNR": 10.0
        }
    }]
}
```

### 使用优先级

1. **音色描述**: 文字描述音色特征，优先级最高
2. **参考音频**: 上传音频克隆声音
3. **内置音色 IP**: 使用内置音色（如"灵小甄"）
4. **控制参数**: 情感/方言/风格等作为补充

## 常见问题

### 模型加载失败

- 确认模型文件在 `./models/Ming-omni-tts-0.5B`
- 检查 CUDA: `docker compose run --rm ming-omni-tts python -c "import torch; print(torch.cuda.is_available())"`

### GPU 内存不足

- 减小 `max_decode_steps` 参数
- 使用 float16 替代 bfloat16（修改 webui.py 中的 `torch_dtype`）

### 依赖问题

- 始终使用 Docker 环境运行
- 避免在宿主机直接安装依赖

### 导入 MingAudio 类

**重要**: 导入 MingAudio 类时，请使用：

```python
# 正确
from webui import MingAudio

# 错误（会启动整个 webui）
# from cookbooks.test import MingAudio
```
