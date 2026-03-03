#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

import os
import sys
import shutil
import uuid
import json

import torch
import torchaudio
from flask import Flask, request, send_file
from flask_cors import CORS
from loguru import logger
from pypinyin import lazy_pinyin
import opencc

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from spkemb_extractor import SpkembExtractor
from inference import generate_speech as _generate_speech

OUTPUT_DIR = "./output"
CONFIG_DIR = "./saved_configs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

opencc_converter = opencc.OpenCC("s2t")


def get_pinyin(text):
    if not text:
        return ""
    return "".join(lazy_pinyin(text))


def get_pinyin_initials(text):
    if not text:
        return ""
    pinyin_list = lazy_pinyin(text)
    return "".join([p[0] if p else "" for p in pinyin_list])


def copy_audio_to_config_dir(audio_path, config_name):
    if audio_path is None:
        return None
    config_audio_dir = os.path.join(CONFIG_DIR, config_name, "audio")
    os.makedirs(config_audio_dir, exist_ok=True)
    audio_ext = os.path.splitext(audio_path)[1]
    dest_path = os.path.join(config_audio_dir, f"ref_audio{audio_ext}")
    shutil.copy2(audio_path, dest_path)
    return dest_path


def save_config(
    config_name,
    prompt_audio,
    prompt_text,
    emotion,
    dialect,
    style,
    voice_description,
    speech_speed,
    pitch,
    volume,
    max_decode_steps,
    cfg,
    sigma,
    temperature,
):
    if not config_name or not config_name.strip():
        return "请输入配置名称", False

    config_name = config_name.strip()
    config_path = os.path.join(CONFIG_DIR, config_name)

    is_overwrite = os.path.exists(config_path)
    if is_overwrite:
        shutil.rmtree(config_path)

    os.makedirs(config_path, exist_ok=True)

    copied_audio = copy_audio_to_config_dir(prompt_audio, config_name)

    config_data = {
        "name": config_name,
        "task_type": "TTS",
        "prompt_audio": copied_audio,
        "prompt_text": prompt_text,
        "emotion": emotion,
        "dialect": dialect,
        "style": style,
        "voice_description": voice_description,
        "speech_speed": speech_speed,
        "pitch": pitch,
        "volume": volume,
        "max_decode_steps": max_decode_steps,
        "cfg": cfg,
        "sigma": sigma,
        "temperature": temperature,
    }

    config_file = os.path.join(config_path, "config.json")
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, ensure_ascii=False, indent=2)

    msg = (
        f"配置 '{config_name}' 已覆盖"
        if is_overwrite
        else f"配置 '{config_name}' 保存成功!"
    )
    return msg, True


def get_config_list():
    if not os.path.exists(CONFIG_DIR):
        return []
    configs = []
    for item in os.listdir(CONFIG_DIR):
        item_path = os.path.join(CONFIG_DIR, item)
        if os.path.isdir(item_path):
            config_file = os.path.join(item_path, "config.json")
            if os.path.exists(config_file):
                configs.append(item)
    return sorted(configs)


def load_config(config_name):
    if not config_name:
        return None, "请选择要加载的配置"

    config_path = os.path.join(CONFIG_DIR, config_name, "config.json")
    if not os.path.exists(config_path):
        return None, f"配置 '{config_name}' 不存在"

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    return config_data, "配置加载成功"


def delete_config(config_name):
    if not config_name:
        return "请选择要删除的配置", False

    config_path = os.path.join(CONFIG_DIR, config_name)
    if not os.path.exists(config_path):
        return f"配置 '{config_name}' 不存在", False

    shutil.rmtree(config_path)
    return f"配置 '{config_name}' 已删除", True


def create_api(model):
    app = Flask(__name__)
    CORS(app)

    @app.route("/")
    def handle_request():
        text = request.args.get("text", "")
        speaker = request.args.get("speaker", "小缘")

        config_list = get_config_list()
        default_speaker = config_list[0] if config_list else "小缘"
        config_options = "".join(
            [f'<option value="{c}">{c}</option>' for c in config_list]
        )

        config_list_json = json.dumps(
            [
                {"name": c, "pinyin": get_pinyin(c), "initials": get_pinyin_initials(c)}
                for c in config_list
            ],
            ensure_ascii=False,
        )

        if not text:
            return f"""<!DOCTYPE html>
<html>
<head>
    <title>Ming-Omni-TTS WebUI</title>
    <meta charset="utf-8">
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; background: #f5f5f5; }}
        h1 {{ color: #333; }}
        .form-group {{ margin-bottom: 15px; }}
        label {{ display: block; margin-bottom: 5px; font-weight: bold; }}
        input[type="text"], textarea, select {{ width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; font-size: 14px; }}
        button {{ background: #4CAF50; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }}
        button:hover {{ background: #45a049; }}
        #result {{ margin-top: 20px; }}
        audio {{ width: 100%; margin-top: 10px; }}
        .info {{ background: #e3f2fd; padding: 15px; border-radius: 4px; margin-bottom: 20px; border-left: 4px solid #2196F3; }}
        .container {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .search-dropdown {{ position: absolute; max-height: 200px; overflow-y: auto; background: white; border: 1px solid #ddd; border-radius: 4px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); z-index: 1000; }}
        .search-dropdown div {{ padding: 10px 12px; cursor: pointer; border-bottom: 1px solid #f0f0f0; }}
        .search-dropdown div:last-child {{ border-bottom: none; }}
        .search-dropdown div:hover {{ background: #f5f5f5; }}
        .search-dropdown div.no-result {{ color: #999; cursor: default; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Ming-Omni-TTS 语音合成</h1>
        <div class="info">
            <p><strong>📡 API 调用方式：</strong></p>
            <code id="api-example">GET http://10.10.10.10:7860/?text=要合成的文本&speaker={default_speaker}</code>
        </div>
        <div class="form-group">
            <label>输入文本：</label>
            <textarea id="text" rows="3" placeholder="请输入要合成语音的文本..."></textarea>
        </div>
        <div class="form-group" style="position: relative;">
            <label>说话人配置：</label>
            <input type="text" id="speaker_search" placeholder="搜索说话人配置..." autocomplete="off">
            <div id="speaker_dropdown" class="search-dropdown" style="display: none; width: 100%;"></div>
            <input type="hidden" id="speaker" value="">
        </div>
        <button onclick="generate()">🎵 生成语音</button>
        <div id="result"></div>
    </div>
    <script>
        var configDataList = {config_list_json};
        
        function showSpeakerDropdown(filter) {{
            var dropdown = document.getElementById('speaker_dropdown');
            var input = document.getElementById('speaker_search');
            dropdown.innerHTML = '';
            dropdown.style.display = 'block';
            
            var filterLower = filter.toLowerCase().trim();
            var filterNoSpace = filterLower.replace(/\\s+/g, '');
            
            var matched = configDataList.filter(function(c) {{
                var nameLower = c.name.toLowerCase();
                var pinyinLower = (c.pinyin || '').toLowerCase();
                var initialsLower = (c.initials || '').toLowerCase().replace(/\\s+/g, '');
                
                return nameLower.includes(filterLower) ||
                       pinyinLower.includes(filterLower) ||
                       initialsLower.includes(filterNoSpace);
            }}).map(function(c) {{ return c.name; }});
            
            if (matched.length === 0) {{
                var noResult = document.createElement('div');
                noResult.className = 'no-result';
                noResult.textContent = '无匹配结果';
                dropdown.appendChild(noResult);
                return;
            }}
            
            matched.forEach(function(name) {{
                var div = document.createElement('div');
                div.textContent = name;
                div.onclick = function(e) {{
                    e.stopPropagation();
                    input.value = name;
                    document.getElementById('speaker').value = name;
                    dropdown.style.display = 'none';
                    updateApiExample();
                }};
                dropdown.appendChild(div);
            }});
        }}
        
        document.getElementById('speaker_search').addEventListener('input', function() {{
            var value = this.value;
            if (!value) {{
                document.getElementById('speaker').value = '';
            }}
            if (value) {{
                showSpeakerDropdown(value);
            }} else {{
                showSpeakerDropdown('');
            }}
        }});
        
        document.getElementById('speaker_search').addEventListener('focus', function() {{
            showSpeakerDropdown(this.value);
        }});
        
        document.addEventListener('click', function(e) {{
            if (!e.target.closest('.form-group') || !e.target.closest('#speaker_search')) {{
                document.getElementById('speaker_dropdown').style.display = 'none';
            }}
        }});
        
        // Set default value
        if (configDataList.length > 0) {{
            document.getElementById('speaker').value = configDataList[0].name;
            document.getElementById('speaker_search').value = configDataList[0].name;
            updateApiExample();
        }}
        
        function updateApiExample() {{
            const speaker = document.getElementById('speaker').value;
            const example = 'http://10.10.10.10:7860/?text=要合成的文本&speaker=' + encodeURIComponent(speaker);
            document.getElementById('api-example').textContent = example;
        }}
        
        async function generate() {{
            const text = document.getElementById('text').value;
            const speaker = document.getElementById('speaker').value;
            const result = document.getElementById('result');
            
            if (!text) {{
                result.innerHTML = '<p style="color:red;">请输入文本</p>';
                return;
            }}
            
            result.innerHTML = '<p>⏳ 正在生成...</p>';
            
            try {{
                const url = '/?text=' + encodeURIComponent(text) + '&speaker=' + encodeURIComponent(speaker);
                const response = await fetch(url);
                
                if (!response.ok) {{
                    const error = await response.text();
                    result.innerHTML = '<p style="color:red;">❌ 错误: ' + error + '</p>';
                    return;
                }}
                
                const blob = await response.blob();
                const audioUrl = URL.createObjectURL(blob);
                result.innerHTML = '<audio controls src="' + audioUrl + '"></audio>';
            }} catch (e) {{
                result.innerHTML = '<p style="color:red;">❌ 错误: ' + e.message + '</p>';
            }}
        }}
    </script>
</body>
</html>"""

        logger.info(f"API请求: text='{text[:50]}...' speaker='{speaker}'")

        config_data, msg = load_config(speaker)
        if config_data is None:
            logger.warning(f"配置 '{speaker}' 不存在，使用默认参数")
            config_data = {
                "prompt_audio": None,
                "prompt_text": None,
                "emotion": None,
                "dialect": None,
                "style": None,
                "voice_description": None,
                "speech_speed": 1.0,
                "pitch": 1.0,
                "volume": 1.0,
                "max_decode_steps": 200,
                "cfg": 2.0,
                "sigma": 0.25,
                "temperature": 0.0,
            }

        output_path = os.path.join(OUTPUT_DIR, f"api_{uuid.uuid4().hex}.wav")

        try:
            result = _generate_speech(
                model=model,
                text=text,
                task_type="语音合成 (TTS)",
                prompt_audio=config_data.get("prompt_audio"),
                prompt_text=config_data.get("prompt_text"),
                emotion=config_data.get("emotion"),
                dialect=config_data.get("dialect"),
                style=config_data.get("style"),
                speech_speed=config_data.get("speech_speed", 1.0),
                pitch=config_data.get("pitch", 1.0),
                volume=config_data.get("volume", 1.0),
                max_decode_steps=config_data.get("max_decode_steps", 200),
                cfg=config_data.get("cfg", 2.0),
                sigma=config_data.get("sigma", 0.25),
                temperature=config_data.get("temperature", 0.0),
                voice_description=config_data.get("voice_description"),
                output_path=output_path,
            )

            if result[0] is None:
                return f"生成失败: {result[1]}", 500

            logger.info(f"生成成功: {output_path}")
            return send_file(output_path, mimetype="audio/wav", as_attachment=True)
        except Exception as e:
            logger.error(f"生成失败: {e}")
            return f"生成失败: {str(e)}", 500

    return app


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    from webui import MingAudio
    from werkzeug.serving import run_simple

    model_path = os.environ.get("MODEL_PATH", "./models/Ming-omni-tts-0.5B")
    port = int(os.environ.get("PORT", 7860))

    logger.info(f"Loading model from {model_path}...")

    model = MingAudio(model_path)

    flask_app = create_api(model)

    print(f"\n{'=' * 60}")
    print(f"🎤 Ming-Omni-TTS API 服务已启动!")
    print(f"{'=' * 60}")
    print(f"📡 API 服务: http://localhost:{port}/")
    print(f"   调用示例: http://localhost:{port}/?text=你好世界&speaker=小缘")
    print(f"{'=' * 60}\n")

    run_simple("0.0.0.0", port, flask_app, use_reloader=False, use_debugger=False)
