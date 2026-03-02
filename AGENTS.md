#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

# AGENTS.md - Ming-omni-tts 开发指南

## 项目概述

Ming-omni-tts 是一个高性能的统一音频生成模型，支持语音、音乐和声音的生成与控制。基于 PyTorch 2.6.0 和 CUDA 12.4 开发。

## 环境设置

### Docker 环境

```bash
# 开发环境构建
docker compose -f docker-compose.dev.yml build

# 生产环境构建
docker compose -f docker-compose.prod.yml build
```

### 宿主机要求

- NVIDIA GPU (RTX 3060+ 推荐)
- Docker + NVIDIA Container Toolkit
- 模型文件放在 `./models/Ming-omni-tts-0.5B` 目录

---

## 运行命令

### 开发模式

```bash
# 进入容器交互开发
docker compose -f docker-compose.dev.yml run --rm ming-omni-tts

# 启动 WebUI（开发）
docker compose -f docker-compose.dev.yml run --rm -p 7860:7860 ming-omni-tts python webui.py
```

### 生产模式

```bash
# 启动生产服务（WebUI + API）
docker compose -f docker-compose.prod.yml up -d

# 查看日志
docker compose -f docker-compose.prod.yml logs -f
```

### 运行测试

```bash
# 完整测试
docker compose -f docker-compose.dev.yml run --rm ming-omni-tts python cookbooks/test.py

# 单个测试用例
docker compose -f docker-compose.dev.yml run --rm ming-omni-tts python -c "
import sys; sys.path.insert(0, '.')
from webui import MingAudio
model = MingAudio('./models/Ming-omni-tts-0.5B')
model.speech_generation(
    prompt='Please generate speech based on the following description.\n',
    text='你好世界',
    use_zero_spk_emb=True,
    output_wav_path='output/test.wav'
)"
```

### Lint 检查

```bash
# 语法检查
docker compose -f docker-compose.dev.yml run --rm ming-omni-tts python -m py_compile <file.py>

# ruff 检查（需安装）
docker compose -f docker-compose.dev.yml run --rm ming-omni-tts pip install ruff && ruff check .
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| MODEL_PATH | ./models/Ming-omni-tts-0.5B | 模型路径 |
| PORT | 7860 | 服务端口 |
| LOAD_MODEL | true | 是否加载模型 |

---

## 代码风格指南

### 文件头

```python
#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.
```

### 导入顺序

1. Python 标准库 (os, sys, json, typing)
2. 第三方库 (torch, gradio, flask, loguru)
3. 本地模块

```python
import os
import sys
from typing import Dict, Optional, List

import torch
import gradio as gr
from flask import Flask
from loguru import logger

from modeling_bailingmm import BailingMMNativeForConditionalGeneration
```

### 命名规范

- **类名**: PascalCase (`MingAudio`, `SpkembExtractor`)
- **函数/变量**: snake_case (`seed_everything`, `model_path`)
- **常量**: 全大写 (`MAX_DECODE_STEPS`, `OUTPUT_DIR`)
- **私有方法**: 前缀下划线 (`_extract_spk_embedding`)

### 类型注解

必须为新代码添加类型注解：

```python
def generate_speech(text: str, max_decode_steps: int = 200) -> torch.Tensor:
    ...

class MingAudio:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.device: str = device
        self.model: Optional[torch.nn.Module] = None
```

### 错误处理

使用 loguru，不要使用 print：

```python
from loguru import logger

try:
    result = model.generate(...)
except Exception as e:
    logger.error(f"生成失败: {e}")
    raise
```

### 代码格式化

- 缩进: 4 空格
- 行长度: 最大 120 字符
- 字符串引号: 双引号优先
- 类之间空 2 行，函数之间空 1 行

### 文档字符串

为所有公共方法添加 docstring：

```python
class MingAudio:
    """Ming-Omni-TTS 语音合成模型封装类"""
    
    def speech_generation(
        self,
        prompt: str,
        text: str,
        use_zero_spk_emb: bool = False,
        max_decode_steps: int = 200,
    ) -> torch.Tensor:
        """
        生成语音音频
        
        Args:
            prompt: 提示词
            text: 输入文本
            use_zero_spk_emb: 是否使用零说话人embedding
            max_decode_steps: 最大解码步数
        
        Returns:
            torch.Tensor: 生成的波形
        """
```

---

## 项目结构

```
Ming-omni-tts/
├── webui.py                   # WebUI + MingAudio 类
├── api.py                     # API 服务
├── modeling_bailingmm.py      # 主模型
├── spkemb_extractor.py        # 说话人embedding
├── audio_tokenizer/           # 音频 tokenizer
├── fm/                        # 流匹配模块
├── sentence_manager/          # 文本规范化
├── cookbooks/test.py          # 测试脚本
└── models/                    # 模型文件
```

## 推理类别

| 类别 | Prompt |
|------|--------|
| TTS | "Please generate speech based on the following description.\n" |
| TTA | "Please generate audio events based on given text.\n" |
| BGM | "Please generate music based on the following description.\n" |

## 导入注意

```python
# 正确
from webui import MingAudio

# 错误（会启动整个 webui）
# from cookbooks.test import MingAudio
```
