#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

# AGENTS.md - Ming-omni-tts Development Guide

## Project Overview
Ming-omni-tts is a high-performance unified audio generation model supporting speech, music, and sound generation/control. Built on PyTorch 2.6.0 and CUDA 12.4.

---

## Build & Test Commands

### Local Development
```bash
# Syntax check a single Python file
python -m py_compile <file.py>

# Run linting
pip install ruff && ruff check .

# Run the full test suite
python cookbooks/test.py

# Run a single test case inline
python -c "
import sys; sys.path.insert(0, '.')
from webui import MingAudio
model = MingAudio('./models/Ming-omni-tts-0.5B')
model.speech_generation(
    prompt='Please generate speech based on the following description.\n',
    text='你好世界',
    use_zero_spk_emb=True,
    output_wav_path='output/test.wav'
)"

# Start WebUI / API server
python webui.py
python api.py
```

### Docker Development
```bash
docker compose -f docker-compose.dev.yml build
docker compose -f docker-compose.dev.yml run --rm ming-omni-tts python cookbooks/test.py
docker compose -f docker-compose.dev.yml run --rm ming-omni-tts pip install ruff && ruff check .
```

---

## Code Style Guidelines

### File Header
```python
#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.
```

### Import Order
1. Python standard library (os, sys, json, typing, copy, re)
2. Third-party libraries (torch, torchaudio, transformers, flask, loguru)
3. Local modules (relative imports after sys.path manipulation)

### Naming Conventions
- **Classes**: PascalCase (`MingAudio`, `SpkembExtractor`)
- **Functions/Variables**: snake_case (`seed_everything`, `model_path`)
- **Constants**: UPPER_CASE (`MAX_DECODE_STEPS`, `OUTPUT_DIR`)
- **Private Methods**: Leading underscore (`_extract_spk_embedding`)
- **Instance Variables**: snake_case (`self.device`, `self.model`)

### Type Annotations (required for new code)
```python
def generate_speech(text: str, max_decode_steps: int = 200, cfg: float = 2.0) -> torch.Tensor:
    ...

class MingAudio:
    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.device: str = device
        self.model: Optional[BailingMMNativeForConditionalGeneration] = None
```

### Error Handling
Use loguru, never print:
```python
from loguru import logger

try:
    result = model.generate(...)
except Exception as e:
    logger.error(f"Generation failed: {e}")
    raise
```

### Code Formatting
- Indentation: 4 spaces
- Max line length: 120 characters
- String quotes: Double quotes preferred
- Blank lines: 2 between classes, 1 between functions
- Docstrings: Required for all public methods

### JavaScript/HTML Guidelines
- Keep HTML/CSS/JS in `templates/` directory
- Use `render_template()`, never `render_template_string()`

---

## Project Structure
```
Ming-omni-tts/
├── webui.py                   # WebUI + MingAudio class
├── api.py                     # API service
├── modeling_bailingmm.py      # Main model
├── spkemb_extractor.py        # Speaker embedding
├── common.py                  # Utilities
├── inference.py               # Inference functions
├── templates/                 # Flask templates
├── audio_tokenizer/           # Audio tokenizer
├── fm/                        # Flow matching modules
├── sentence_manager/          # Text normalization
├── cookbooks/test.py          # Test script
└── models/                    # Model files
```

---

## Inference Categories
| Category | Prompt |
|----------|--------|
| TTS | "Please generate speech based on the following description.\n" |
| TTA | "Please generate audio events based on given text.\n" |
| BGM | "Please generate music based on the following description.\n" |

---

## Import Guidelines
```python
# Correct - imports MingAudio class
from webui import MingAudio

# Wrong - loads entire webui including Flask app
# from cookbooks.test import MingAudio
```

---

## Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| MODEL_PATH | ./models/Ming-omni-tts-0.5B | Model path |
| PORT | 7860 | Service port |

---

## Key Classes
### MingAudio (webui.py / cookbooks/test.py)
Main interface: `speech_generation()`, `generation()`, `create_instruction()`

### BailingMMNativeForConditionalGeneration (modeling_bailingmm.py)
HuggingFace-compatible model class for all generation logic.

### SpkembExtractor (spkemb_extractor.py)
Extracts speaker embeddings from audio files.
