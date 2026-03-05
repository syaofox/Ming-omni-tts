#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

import os
import shutil
import json

import opencc
from pypinyin import lazy_pinyin

import tempfile

OUTPUT_DIR = "./output"
CONFIG_DIR = "./saved_configs"
UPLOAD_DIR = tempfile.gettempdir()

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

opencc_converter = opencc.OpenCC("s2t")


def get_pinyin(text: str) -> str:
    if not text:
        return ""
    return "".join(lazy_pinyin(text))


def get_pinyin_initials(text: str) -> str:
    if not text:
        return ""
    pinyin_list = lazy_pinyin(text)
    return "".join([p[0] if p else "" for p in pinyin_list])


def to_traditional(text: str) -> str:
    if not text:
        return text
    return opencc_converter.convert(text)


def to_simplified(text: str) -> str:
    if not text:
        return text
    return opencc.convert(text)


def copy_audio_to_config_dir(audio_path: str, config_name: str):
    if audio_path is None or audio_path == "":
        return None

    if not os.path.isabs(audio_path) and audio_path.startswith("./"):
        audio_path = os.path.abspath(audio_path)

    if not os.path.exists(audio_path):
        return None

    config_audio_dir = os.path.join(CONFIG_DIR, config_name, "audio")
    os.makedirs(config_audio_dir, exist_ok=True)
    audio_ext = os.path.splitext(audio_path)[1]
    dest_path = os.path.join(config_audio_dir, f"ref_audio{audio_ext}")
    shutil.copy2(audio_path, dest_path)
    return dest_path


def save_config(
    config_name: str,
    task_type: str,
    prompt_audio,
    prompt_text=None,
    emotion=None,
    dialect=None,
    style=None,
    voice_description=None,
    speech_speed=None,
    pitch=None,
    volume=None,
    max_decode_steps=None,
    cfg=None,
    sigma=None,
    temperature=None,
    ip=None,
    instruct_type=None,
):
    from loguru import logger

    logger.info(
        f"save_config called: config_name={config_name}, task_type={task_type}, prompt_audio={prompt_audio}"
    )

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
        "task_type": task_type,
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
        "ip": ip,
        "instruct_type": instruct_type,
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
                with open(config_file, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                configs.append(
                    {"name": item, "task_type": config_data.get("task_type", "TTS")}
                )
    return sorted(configs, key=lambda x: x["name"])


def load_config(config_name: str):
    if not config_name:
        return None, "请选择要加载的配置"

    config_path = os.path.join(CONFIG_DIR, config_name, "config.json")
    if not os.path.exists(config_path):
        return None, f"配置 '{config_name}' 不存在"

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    return config_data, "配置加载成功"


def delete_config(config_name: str):
    if not config_name:
        return "请选择要删除的配置", False

    config_path = os.path.join(CONFIG_DIR, config_name)
    if not os.path.exists(config_path):
        return f"配置 '{config_name}' 不存在", False

    shutil.rmtree(config_path)
    return f"配置 '{config_name}' 已删除", True
