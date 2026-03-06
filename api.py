#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

import os
import random
import sys
import uuid

import torch
from flask import Flask, request, send_file, render_template, jsonify
from flask_cors import CORS
from loguru import logger

scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
sys.path.insert(0, scripts_dir)
from generate_http_tts_config import generate_http_tts_config

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common import (
    OUTPUT_DIR,
    CONFIG_DIR,
    get_pinyin,
    get_pinyin_initials,
    load_config,
    get_config_list,
)

from inference import generate_speech as _generate_speech


def create_api(model):
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    CORS(app)

    @app.route("/")
    def handle_request():
        text = request.args.get("text", "")
        speaker = request.args.get("speaker", "小缘")

        config_list = get_config_list()
        default_speaker = config_list[0]["name"] if config_list else "小缘"

        config_list_for_template = []
        if config_list:
            config_list_for_template = [
                {
                    "name": c["name"],
                    "value": c["name"],
                    "pinyin": get_pinyin(c["name"]),
                    "initials": get_pinyin_initials(c["name"]),
                }
                for c in config_list
            ]

        if not text:
            host = request.host
            if ":" in host:
                server_ip = host.split(":")[0]
            else:
                server_ip = host
            return render_template(
                "api.html",
                config_list=config_list_for_template,
                default_speaker=default_speaker,
                server_ip=server_ip,
            )

        logger.info(f"API请求: text='{text[:50]}...' speaker='{speaker}'")

        config_data, msg = load_config(speaker)
        if config_data is None:
            logger.warning(f"配置 '{speaker}' 不存在，使用默认参数")
            config_data = {
                "task_type": "Instruct TTS",
                "prompt_audio": None,
                "prompt_text": None,
                "emotion": None,
                "dialect": None,
                "style": None,
                "ip": None,
                "speech_speed": 1.0,
                "pitch": 1.0,
                "volume": 1.0,
                "max_decode_steps": 200,
                "cfg": 2.0,
                "sigma": 0.25,
                "temperature": 0.0,
            }

        # 根据配置中的 task_type 映射到 inference 需要的 task_type
        task_type_map = {
            "Instruct TTS": "Instruct TTS",
            "Zero-shot TTS": "零样本语音合成 (Zero-shot TTS)",
            "Podcast": "Podcast",
            "BGM Generation": "BGM Generation",
            "Sound Effects (TTA)": "声音事件 (TTA)",
        }
        task_type = task_type_map.get(
            config_data.get("task_type", "Instruct TTS"), "Instruct TTS"
        )

        # 使用 or 处理 None 值，而非 .get() 的默认值
        speech_speed = config_data.get("speech_speed") or 1.0
        pitch = config_data.get("pitch") or 1.0
        volume = config_data.get("volume") or 1.0
        max_decode_steps = config_data.get("max_decode_steps") or 200
        cfg = config_data.get("cfg") or 2.0
        sigma = config_data.get("sigma") or 0.25
        temperature = config_data.get("temperature") or 0.0

        output_path = os.path.join(OUTPUT_DIR, f"api_{uuid.uuid4().hex}.wav")

        try:
            result = _generate_speech(
                model=model,
                text=text,
                task_type=task_type,
                prompt_audio=config_data.get("prompt_audio"),
                prompt_text=config_data.get("prompt_text"),
                emotion=config_data.get("emotion"),
                dialect=config_data.get("dialect"),
                style=config_data.get("style"),
                speech_speed=speech_speed,
                pitch=pitch,
                volume=volume,
                max_decode_steps=max_decode_steps,
                cfg=cfg,
                sigma=sigma,
                temperature=temperature,
                ip=config_data.get("ip"),
                output_path=output_path,
                seed=random.randint(0, 2**32 - 1),
                bgm=None,
                podcast_task=(task_type == "Podcast"),
            )

            if result[0] is None:
                return f"生成失败: {result[1]}", 500

            logger.info(f"生成成功: {output_path}")
            return send_file(output_path, mimetype="audio/wav", as_attachment=True)
        except Exception as e:
            logger.error(f"生成失败: {e}")
            return f"生成失败: {str(e)}", 500

    @app.route("/api/generate-config", methods=["POST"])
    def generate_config():
        data = request.get_json() or {}
        ip = data.get("ip", "10.10.10.10")
        port = data.get("port", 7861)

        try:
            output_path = os.path.join(current_dir, "saved_configs", "httpTts.json")
            configs = generate_http_tts_config(
                ip=ip,
                port=port,
                output_path=output_path,
            )
            return jsonify(
                {"success": True, "count": len(configs), "preview": configs[:3]}
            )
        except Exception as e:
            logger.error(f"生成配置失败: {e}")
            return jsonify({"success": False, "error": str(e)}), 500

    @app.route("/api/download-config")
    def download_config():
        config_path = os.path.join(current_dir, "saved_configs", "httpTts.json")
        return send_file(
            config_path,
            mimetype="application/json",
            as_attachment=True,
            download_name="httpTts.json",
        )

    return app


if __name__ == "__main__":
    import warnings
    import json

    warnings.filterwarnings("ignore")

    from webui import MingAudio
    from werkzeug.serving import run_simple

    model_path = os.environ.get("MODEL_PATH", "./models/Ming-omni-tts-0.5B")
    port = int(os.environ.get("PORT", 7860))

    logger.info(f"Loading model from {model_path}...")

    model = MingAudio(model_path)

    flask_app = create_api(model)

    print(f"\n{'=' * 60}")
    print(f"Ming-Omni-TTS API 服务已启动!")
    print(f"{'=' * 60}")
    print(f"API 服务: http://localhost:{port}/")
    print(f"   调用示例: http://localhost:{port}/?text=你好世界&speaker=小缘")
    print(f"{'=' * 60}\n")

    run_simple("0.0.0.0", port, flask_app, use_reloader=False, use_debugger=False)
