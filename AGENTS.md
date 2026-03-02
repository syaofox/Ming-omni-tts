# AGENTS.md - Ming-omni-tts 开发指南

## 项目概述

Ming-omni-tts 是一个高性能的统一音频生成模型，支持语音、音乐和声音的生成与控制。基于 PyTorch 2.6.0 和 CUDA 12.4 开发。

## 环境设置

### Docker 环境（推荐）

本项目使用 Docker 部署，已配置好 CUDA 12.4 和所有依赖。

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

### 构建命令

```bash
# 构建 Docker 镜像
docker compose build

# 重新构建（不带缓存）
docker compose build --no-cache
```

### 运行命令

```bash
# 运行完整测试（所有示例）
docker compose run --rm ming-omni-tts python cookbooks/test.py

# 运行单个测试用例 - 编辑 test.py 注释不需要的测试
docker compose run --rm ming-omni-tts python -c "
import sys
sys.path.insert(0, '.')
# 只运行 TTA 测试
exec('''
from cookbooks.test import MingAudio
model = MingAudio(\"./models/Ming-omni-tts-0.5B\")
decode_args = {\"max_decode_steps\": 200, \"cfg\": 4.5, \"sigma\": 0.3, \"temperature\": 2.5}
messages = {\"prompt\": \"Please generate audio events based on given text.\n\", \"text\": \"Thunder and a gentle rain\"}
model.speech_generation(**messages, **decode_args, output_wav_path=\"output/tta.wav\")
''')
"
```

### Lint 命令

项目暂无 formal lint 配置，可使用以下工具：

```bash
# Python 语法检查
docker compose run --rm ming-omni-tts python -m py_compile <file.py>

# ruff (需安装)
docker compose run --rm ming-omni-tts pip install ruff && ruff check .

# black 格式化
docker compose run --rm ming-omni-tts pip install black && black --check .
```

## 代码风格指南

### 文件头

```python
#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.
```

### 导入顺序

1. Python 标准库 (os, sys, json, etc.)
2. 第三方库 (torch, transformers, loguru)
3. 本地模块 (from .xxx import, from cookbooks.xxx)

```python
# 标准库
import os
import sys
import json
from typing import Dict, Optional, List

# 第三方库
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from loguru import logger

# 本地模块
from modeling_bailingmm import BailingMMNativeForConditionalGeneration
from sentence_manager.sentence_manager import SentenceNormalizer
```

### 命名规范

- **类名**: PascalCase (如 `MingAudio`, `SpkembExtractor`)
- **函数/变量**: snake_case (如 `seed_everything`, `model_path`)
- **常量**: 全大写 snake_case (如 `MAX_DECODE_STEPS`)
- **私有方法/变量**: 前缀下划线 (如 `_extract_spk_embedding`)

### 类型注解

推荐为新代码添加类型注解：

```python
# 函数类型注解
def generate_speech(
    text: str,
    prompt_wav: Optional[str] = None,
    max_decode_steps: int = 200
) -> torch.Tensor:

# 类属性类型注解
class MingAudio:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.device: str = device
        self.model = None
        self.sample_rate: int = 0
```

### 错误处理

使用 loguru 进行日志记录：

```python
from loguru import logger

# 捕获异常并记录
try:
    result = model.generate(...)
except Exception as e:
    logger.error(f"生成失败: {e}")
    raise

# 警告信息
logger.warning("GPU 内存不足，切换到 CPU 模式")
```

### 代码格式化

- 缩进: 4 空格
- 行长度: 推荐 100 字符以内
- 字符串引号: 双引号优先

```python
# 推荐
instruction = {"情感": "高兴", "语速": "快速"}
prompt = "Please generate speech based on the following description.\n"

# 不推荐
instruction = {'情感': '高兴'}  # 单引号
```

### 文档字符串

为公共方法添加文档字符串：

```python
class MingAudio:
    def speech_generation(self, prompt, text, **kwargs):
        """
        生成语音音频
        
        Args:
            prompt: 提示词
            text: 输入文本
            prompt_wav_path: 参考音频路径
            max_decode_steps: 最大解码步数
        
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
├── audio_tokenizer/          # 音频 tokenizer
│   ├── modeling_audio_vae.py
│   └── audio_encoder.py
├── fm/                       # 流匹配模块
│   ├── dit.py
│   └── flowloss.py
├── sentence_manager/         # 文本规范化
├── cookbooks/
│   └── test.py              # 测试/演示脚本
├── docker/
│   └── ming_uniaudio.dockerfile
└── models/                  # 模型文件目录
```

## 常见问题

### 模型加载失败

- 确认模型文件在 `./models/Ming-omni-tts-0.5B`
- 检查 CUDA 是否可用: `docker compose run --rm ming-omni-tts python -c "import torch; print(torch.cuda.is_available())"`

### GPU 内存不足

- 减小 `max_decode_steps` 参数
- 使用 float16 替代 bfloat16（修改 test.py 中的 `torch_dtype`）

### 依赖问题

- 始终使用 Docker 环境运行
- 避免在宿主机直接安装依赖
