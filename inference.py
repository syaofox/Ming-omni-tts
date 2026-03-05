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
    ip=None,
):
    instruction = {}
    if ip and ip.strip():
        instruction["IP"] = ip.strip()
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
    if task_type == "语音合成 (TTS)" or task_type == "Instruct TTS":
        return "Please generate speech based on the following description.\n"
    elif task_type == "零样本语音合成 (Zero-shot TTS)":
        return "Please generate speech based on the following description.\n"
    elif task_type == "声音事件 (TTA)":
        return "Please generate audio events based on given text.\n"
    elif task_type == "背景音乐 (BGM)" or task_type == "BGM Generation":
        return "Please generate music based on the following description.\n"
    elif task_type == "Podcast":
        return "Please generate speech based on the following description.\n"
    elif task_type == "Speech with BGM":
        return "Please generate speech with background music based on the following description.\n"
    else:
        return "Please generate speech based on the following description.\n"


def preprocess_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text_list = [t.strip() for t in text.split("\n") if t.strip()]
    return text_list


def parse_podcast_dialogues(text):
    """
    解析对话文本，返回按原文顺序的句子列表
    每个元素: (speaker_index, sentence_text)
    speaker_index 从 1 开始 (speaker_1 -> 0, speaker_2 -> 1, ...)
    """
    import re

    sentences = []
    lines = text.replace("\r\n", "\n").replace("\r", "\n").strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # 匹配 speaker_X: 格式
        match = re.match(r"^speaker_(\d+):\s*(.+)$", line, re.IGNORECASE)
        if match:
            speaker_index = int(match.group(1)) - 1  # 转换为 0-based
            sentence_text = match.group(2).strip()
            if sentence_text:
                sentences.append((speaker_index, sentence_text))

    return sentences


def generate_podcast(
    model,
    text,
    prompt_audio_list,
    max_decode_steps,
    cfg,
    sigma,
    temperature,
    output_path,
    seed,
):
    """
    分句生成 Podcast 并合并
    """
    import re
    import torch
    import torchaudio

    # 解析对话
    sentences = parse_podcast_dialogues(text)
    if not sentences:
        return None, "无法解析对话文本，请使用 speaker_1:, speaker_2: 等格式"

    # 按说话人分组句子
    speaker_sentences = {}
    for speaker_idx, sentence in sentences:
        if speaker_idx not in speaker_sentences:
            speaker_sentences[speaker_idx] = []
        speaker_sentences[speaker_idx].append(sentence)

    # 检查参考音频数量
    num_speakers = len(speaker_sentences)
    if num_speakers > len(prompt_audio_list):
        return (
            None,
            f"对话中有 {num_speakers} 个说话人，但只提供了 {len(prompt_audio_list)} 个参考音频",
        )

    # 预处理每个说话人的参考音频
    prompt_waveforms = []
    spk_embs = []

    for i in range(num_speakers):
        prompt_wav_path = prompt_audio_list[i]
        waveform, sr = torchaudio.load(prompt_wav_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != model.sample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=model.sample_rate
            )(waveform)
        prompt_waveforms.append(waveform)

        # 提取 speaker embedding
        waveform_16k = torchaudio.transforms.Resample(
            orig_freq=model.sample_rate, new_freq=16000
        )(waveform)
        spk_emb = model.spkemb_extractor(waveform_16k)
        spk_embs.append(spk_emb)

    # 生成每个说话人的句子
    generated_waveforms = []
    prompt = get_prompt_by_task_type("Podcast")

    for speaker_idx, sentence in sentences:
        spk_emb = spk_embs[speaker_idx]

        waveform = model.speech_generation(
            prompt=prompt,
            text=sentence,
            use_spk_emb=True,
            use_zero_spk_emb=False,
            instruction=None,
            prompt_wav_path=prompt_audio_list[speaker_idx],
            prompt_text=None,
            max_decode_steps=max_decode_steps,
            cfg=cfg,
            sigma=sigma,
            temperature=temperature,
            output_wav_path=None,
            seed=seed,
        )
        generated_waveforms.append(waveform)

    # 合并所有音频
    final_waveform = torch.cat(generated_waveforms, dim=-1)

    # 保存
    if output_path:
        import os

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        torchaudio.save(output_path, final_waveform, sample_rate=model.sample_rate)

    return output_path, f"生成成功! (共 {len(sentences)} 句, {num_speakers} 人)"


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
    ip=None,
    output_path=None,
    seed=None,
    bgm=None,
    podcast_task=False,
):
    if not prompt_text:
        prompt_text = None

    if model is None:
        return None, "请先加载模型"

    if not text.strip() and task_type != "Podcast":
        return None, "请输入文本内容"

    # 处理 Podcast - 不按行分割文本
    if task_type == "Podcast":
        text_list = [text]  # Podcast 使用原始文本格式
    else:
        text_list = preprocess_text(text)
        if not text_list:
            return None, "请输入文本内容"

    use_spk_emb = prompt_audio is not None
    use_zero_spk_emb = prompt_audio is None

    if task_type == "零样本语音合成 (Zero-shot TTS)":
        use_spk_emb = prompt_audio is not None
        use_zero_spk_emb = not use_spk_emb
    elif task_type == "Instruct TTS":
        if prompt_audio is not None:
            use_spk_emb = True
            use_zero_spk_emb = False
        else:
            use_spk_emb = False
            use_zero_spk_emb = True

    if task_type in ["声音事件 (TTA)", "背景音乐 (BGM)", "BGM Generation"]:
        use_spk_emb = False
        use_zero_spk_emb = False

    # 处理 Podcast - 需要多说话人
    if task_type == "Podcast":
        use_spk_emb = True
        use_zero_spk_emb = False

    # 处理 Speech with BGM - 使用 bgm 参数
    if task_type == "Speech with BGM":
        use_spk_emb = True
        use_zero_spk_emb = False

    # 构建 instruction
    if task_type == "Speech with BGM" and bgm:
        instruction = {"BGM": bgm}
    else:
        instruction = build_instruction(
            voice_description=voice_description,
            emotion=emotion,
            dialect=dialect,
            style=style,
            speech_speed=speech_speed,
            pitch=pitch,
            volume=volume,
            ip=ip,
        )

    prompt = get_prompt_by_task_type(task_type)

    # Podcast 使用单独的分句生成逻辑
    if task_type == "Podcast":
        try:
            return generate_podcast(
                model=model,
                text=text,
                prompt_audio_list=prompt_audio,
                max_decode_steps=max_decode_steps,
                cfg=cfg,
                sigma=sigma,
                temperature=temperature,
                output_path=output_path,
                seed=seed,
            )
        except Exception as e:
            from loguru import logger

            logger.error(f"Podcast generation failed: {e}")
            return None, f"生成失败: {str(e)}"

    # 对于 Podcast，prompt_text 和 text 相同，但模型内部会重复
    # 所以这里不传 prompt_text，让模型只使用 text
    if task_type == "Podcast":
        use_prompt_text = None
    else:
        use_prompt_text = prompt_text if (prompt_audio and prompt_text) else None

    try:
        if len(text_list) == 1:
            waveform = model.speech_generation(
                prompt=prompt,
                text=text_list[0],
                use_spk_emb=use_spk_emb,
                use_zero_spk_emb=use_zero_spk_emb,
                instruction=instruction,
                prompt_wav_path=prompt_audio,
                prompt_text=use_prompt_text,
                max_decode_steps=max_decode_steps,
                cfg=cfg,
                sigma=sigma,
                temperature=temperature,
                output_wav_path=output_path,
                seed=seed,
            )
        else:
            waveform = model.speech_generation_batch(
                prompt=prompt,
                text_list=text_list,
                use_spk_emb=use_spk_emb,
                use_zero_spk_emb=use_zero_spk_emb,
                instruction=instruction,
                prompt_wav_path=prompt_audio,
                prompt_text=use_prompt_text,
                max_decode_steps=max_decode_steps,
                cfg=cfg,
                sigma=sigma,
                temperature=temperature,
                output_wav_path=output_path,
                seed=seed,
            )
        return output_path, f"生成成功! (共 {len(text_list)} 段)"
    except Exception as e:
        from loguru import logger

        logger.error(f"Generation failed: {e}")
        return None, f"生成失败: {str(e)}"
