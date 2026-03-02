#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

import os
import json


def build_instruction(
    voice_description,
    emotion,
    dialect,
    style,
    speech_speed=None,
    pitch=None,
    volume=None,
):
    instruction = {}
    if voice_description and voice_description.strip():
        instruction["音色描述"] = voice_description.strip()
        if emotion and emotion != "无":
            instruction["情感"] = emotion
        if dialect and dialect != "无":
            instruction["方言"] = dialect
        if style and style != "无":
            instruction["风格"] = style
        if speech_speed and speech_speed != 1.0:
            instruction["语速"] = speech_speed
        if pitch and pitch != 1.0:
            instruction["基频"] = pitch
        if volume and volume != 1.0:
            instruction["音量"] = volume
    else:
        if emotion and emotion != "无":
            instruction["情感"] = emotion
        if dialect and dialect != "无":
            instruction["方言"] = dialect
        if style and style != "无":
            instruction["风格"] = style
        if speech_speed and speech_speed != 1.0:
            instruction["语速"] = speech_speed
        if pitch and pitch != 1.0:
            instruction["基频"] = pitch
        if volume and volume != 1.0:
            instruction["音量"] = volume
    return instruction if instruction else None


def get_prompt_by_task_type(task_type):
    if task_type == "语音合成 (TTS)":
        return "Please generate speech based on the following description.\n"
    elif task_type == "声音事件 (TTA)":
        return "Please generate audio events based on given text.\n"
    elif task_type == "背景音乐 (BGM)":
        return "Please generate music based on the following description.\n"
    else:
        return "Please generate speech based on the following description.\n"


def preprocess_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text_list = [t.strip() for t in text.split("\n") if t.strip()]
    return text_list


def generate_speech(
    model,
    text,
    task_type="语音合成 (TTS)",
    prompt_audio=None,
    prompt_text=None,
    emotion=None,
    dialect=None,
    style=None,
    speech_speed=1.0,
    pitch=1.0,
    volume=1.0,
    max_decode_steps=200,
    cfg=2.0,
    sigma=0.25,
    temperature=0.0,
    voice_description=None,
    output_path=None,
):
    if not prompt_text:
        prompt_text = None

    if model is None:
        return None, "请先加载模型"

    if not text.strip():
        return None, "请输入文本内容"

    text_list = preprocess_text(text)
    if not text_list:
        return None, "请输入文本内容"

    use_spk_emb = prompt_audio is not None
    use_zero_spk_emb = prompt_audio is None

    instruction = build_instruction(
        voice_description=voice_description,
        emotion=emotion,
        dialect=dialect,
        style=style,
        speech_speed=speech_speed,
        pitch=pitch,
        volume=volume,
    )

    prompt = get_prompt_by_task_type(task_type)

    try:
        if len(text_list) == 1:
            waveform = model.speech_generation(
                prompt=prompt,
                text=text_list[0],
                use_spk_emb=use_spk_emb,
                use_zero_spk_emb=use_zero_spk_emb,
                instruction=instruction,
                prompt_wav_path=prompt_audio,
                prompt_text=prompt_text if (prompt_audio and prompt_text) else None,
                max_decode_steps=max_decode_steps,
                cfg=cfg,
                sigma=sigma,
                temperature=temperature,
                output_wav_path=output_path,
            )
        else:
            waveform = model.speech_generation_batch(
                prompt=prompt,
                text_list=text_list,
                use_spk_emb=use_spk_emb,
                use_zero_spk_emb=use_zero_spk_emb,
                instruction=instruction,
                prompt_wav_path=prompt_audio,
                prompt_text=prompt_text if (prompt_audio and prompt_text) else None,
                max_decode_steps=max_decode_steps,
                cfg=cfg,
                sigma=sigma,
                temperature=temperature,
                output_wav_path=output_path,
            )
        return output_path, f"生成成功! (共 {len(text_list)} 段)"
    except Exception as e:
        from loguru import logger

        logger.error(f"Generation failed: {e}")
        return None, f"生成失败: {str(e)}"
