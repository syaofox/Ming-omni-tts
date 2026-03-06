#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

# AGENTS.md - Ming-omni-tts 开发指南

**注意：本项目为本地无环境部署，忽略所有 LSP 错误。**

---

## 项目概述
Ming-omni-tts 是一个高性能的统一音频生成模型，支持语音、音乐和声音的生成/控制。基于 PyTorch 2.6.0 和 CUDA 12.4 构建。

---

## 构建与测试命令

### Docker 开发（必须使用）
```bash
# 构建 Docker 镜像
docker compose -f docker-compose.dev.yml build

# 运行完整测试套件
docker compose -f docker-compose.dev.yml run --rm ming-omni-tts python cookbooks/test.py

# 运行代码检查
docker compose -f docker-compose.dev.yml run --rm ming-omni-tts pip install ruff && ruff check .

# 启动 WebUI 服务（端口 7860）
docker compose -f docker-compose.dev.yml run --rm -p 7860:7860 ming-omni-tts python webui.py

# 启动 API 服务（端口 7860）
docker compose -f docker-compose.dev.yml run --rm -p 7860:7860 ming-omni-tts python api.py
```

### 本地开发（仅用于语法检查）
```bash
# 语法检查单个 Python 文件
python -m py_compile <file.py>
```

---

## 代码风格规范

### 文件头
```python
#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.
```

### 导入顺序
1. Python 标准库 (os, sys, json, typing, copy, re)
2. 第三方库 (torch, torchaudio, transformers, flask, loguru)
3. 本地模块 (sys.path 后的相对导入)

### 命名规范
- **类**: PascalCase (`MingAudio`, `SpkembExtractor`)
- **函数/变量**: snake_case (`seed_everything`, `model_path`)
- **常量**: UPPER_CASE (`MAX_DECODE_STEPS`, `OUTPUT_DIR`)
- **私有方法**: 前导下划线 (`_extract_spk_embedding`)
- **实例变量**: snake_case (`self.device`, `self.model`)

### 类型注解（新增代码必须使用）
```python
def generate_speech(text: str, max_decode_steps: int = 200, cfg: float = 2.0) -> torch.Tensor:
    ...

class MingAudio:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.device: str = device
        self.model: Optional[BailingMMNativeForConditionalGeneration] = None
```

### 错误处理
使用 loguru，禁止使用 print：
```python
from loguru import logger

try:
    result = model.generate(...)
except Exception as e:
    logger.error(f"Generation failed: {e}")
    raise
```

### 代码格式
- 缩进：4 空格
- 最大行长度：120 字符
- 字符串引号：优先使用双引号
- 空行：类之间 2 行，函数之间 1 行
- 文档字符串：所有公开方法必须包含

### JavaScript/HTML 规范
- HTML/CSS/JS 放在 `templates/` 目录
- 使用 `render_template()`，禁止使用 `render_template_string()`

---

## 项目结构
```
Ming-omni-tts/
├── webui.py                   # WebUI + MingAudio 类
├── api.py                     # API 服务
├── modeling_bailingmm.py      # 主模型
├── spkemb_extractor.py        # 说话人embedding提取
├── common.py                  # 工具函数
├── inference.py               # 推理函数
├── templates/                 # Flask 模板
├── audio_tokenizer/           # 音频分词器
├── fm/                        # 流匹配模块
├── sentence_manager/          # 文本规范化
├── cookbooks/test.py          # 测试脚本
└── models/                    # 模型文件
```

---

## 推理类别
| 类别 | Prompt |
|----------|--------|
| TTS | "Please generate speech based on the following description.\n" |
| TTA | "Please generate audio events based on given text.\n" |
| BGM | "Please generate music based on the following description.\n" |

---

## 导入规范
```python
# 正确 - 仅导入 MingAudio 类
from webui import MingAudio

# 错误 - 会加载整个 webui 包括 Flask 应用
# from cookbooks.test import MingAudio
```

---

## 环境变量
| 变量 | 默认值 | 描述 |
|----------|---------|-------------|
| MODEL_PATH | ./models/Ming-omni-tts-0.5B | 模型路径 |
| PORT | 7860 | 服务端口 |

---

## 核心类
### MingAudio (webui.py / cookbooks/test.py)
主要接口：`speech_generation()`, `generation()`, `create_instruction()`

### BailingMMNativeForConditionalGeneration (modeling_bailingmm.py)
HuggingFace 兼容的模型类，包含所有生成逻辑。

### SpkembExtractor (spkemb_extractor.py)
从音频文件中提取说话人 embedding。
