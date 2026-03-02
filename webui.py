#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

import os
import sys
import warnings
import torch
import torchaudio
from transformers import AutoTokenizer
import gradio as gr
from flask import Flask, request, send_file
from flask_cors import CORS
import io
import uuid
from loguru import logger

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from modeling_bailingmm import BailingMMNativeForConditionalGeneration
from sentence_manager.sentence_manager import SentenceNormalizer
from spkemb_extractor import SpkembExtractor

warnings.filterwarnings("ignore")


class MingAudio:
    def __init__(self, model_path, device="cuda:0"):
        self.device = device
        self.model = BailingMMNativeForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.model = self.model.eval().to(torch.bfloat16).to(self.device)

        if self.model.model_type == "dense":
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(".", trust_remote_code=True)
        self.model.tokenizer = self.tokenizer
        self.sample_rate = self.model.config.audio_tokenizer_config.sample_rate
        self.patch_size = self.model.config.ditar_config["patch_size"]
        self.normalizer = self.init_tn_normalizer(tokenizer=self.tokenizer)

        local_model_path = model_path if os.path.isdir(model_path) else model_path
        self.spkemb_extractor = SpkembExtractor(f"{local_model_path}/campplus.onnx")

    def init_tn_normalizer(self, config_file_path=None, tokenizer=None):
        if config_file_path is None:
            default_config_path = "sentence_manager/default_config.yaml"
            config_file_path = default_config_path

        with open(config_file_path, "r") as f:
            self.sentence_manager_config = {}

        return SentenceNormalizer(self.sentence_manager_config.get("text_norm", {}))

    def create_instruction(self, user_input: dict):
        base_caption_template = {
            "audio_sequence": [
                {
                    "序号": 1,
                    "说话人": "speaker_1",
                    "音色描述": None,
                    "方言": None,
                    "风格": None,
                    "语速": None,
                    "基频": None,
                    "音量": None,
                    "情感": None,
                    "BGM": {
                        "Genre": None,
                        "Mood": None,
                        "Instrument": None,
                        "Theme": None,
                        "ENV": None,
                        "SNR": None,
                    },
                    "IP": None,
                }
            ]
        }
        import copy

        new_caption = copy.deepcopy(base_caption_template)
        target_item_dict = new_caption["audio_sequence"][0]

        for key, value in user_input.items():
            if key in target_item_dict:
                target_item_dict[key] = value

        return new_caption

    def pad_waveform(self, waveform):
        pad_align = int(1 / 12.5 * self.patch_size * self.sample_rate)
        new_len = (waveform.size(-1) + pad_align - 1) // pad_align * pad_align
        if new_len != waveform.size(1):
            new_wav = torch.zeros(
                1, new_len, dtype=waveform.dtype, device=waveform.device
            )
            new_wav[:, : waveform.size(1)] = waveform.clone()
            waveform = new_wav
        return waveform

    def preprocess_one_prompt_wav(self, waveform_path, use_spk_emb):
        if waveform_path is None:
            return None, None

        waveform, sr = torchaudio.load(waveform_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform1 = waveform.clone()
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.sample_rate
            )(waveform)

        if use_spk_emb:
            waveform1 = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(
                waveform1
            )
            spk_emb = self.spkemb_extractor(waveform1)
        else:
            spk_emb = None
        return waveform, spk_emb

    def speech_generation(
        self,
        prompt,
        text,
        use_spk_emb=False,
        use_zero_spk_emb=False,
        instruction=None,
        prompt_wav_path=None,
        prompt_text=None,
        max_decode_steps=200,
        cfg=2.0,
        sigma=0.25,
        temperature=0,
        output_wav_path=None,
    ):
        if prompt_wav_path is None:
            prompt_waveform, prompt_text, spk_emb = None, None, None
            if use_zero_spk_emb:
                spk_emb = [
                    torch.zeros(1, 192, device=self.device, dtype=torch.bfloat16)
                ]
        else:
            paths = (
                prompt_wav_path
                if isinstance(prompt_wav_path, list)
                else [prompt_wav_path]
            )
            processed_prompts = [
                self.preprocess_one_prompt_wav(p, use_spk_emb) for p in paths
            ]
            waveforms_list, spk_emb = zip(*processed_prompts)
            prompt_waveform = torch.cat(waveforms_list, dim=-1)
            prompt_waveform = self.pad_waveform(prompt_waveform)
            spk_emb = list(spk_emb)
            if all([x is None for x in spk_emb]):
                spk_emb = None

        if instruction is not None:
            import json

            instruction = self.create_instruction(instruction)
            instruction = json.dumps(instruction, ensure_ascii=False)

        waveform = self.model.generate(
            prompt=prompt,
            text=text,
            spk_emb=spk_emb,
            instruction=instruction,
            prompt_waveform=prompt_waveform,
            prompt_text=prompt_text,
            max_decode_steps=max_decode_steps,
            cfg=cfg,
            sigma=sigma,
            temperature=temperature,
            use_zero_spk_emb=use_zero_spk_emb,
        )
        if output_wav_path is not None:
            output_dir = os.path.dirname(output_wav_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            torchaudio.save(output_wav_path, waveform, sample_rate=self.sample_rate)
        return waveform

    def speech_generation_batch(
        self,
        prompt,
        text_list: list,
        use_spk_emb=False,
        use_zero_spk_emb=False,
        instruction=None,
        prompt_wav_path=None,
        prompt_text=None,
        max_decode_steps=200,
        cfg=2.0,
        sigma=0.25,
        temperature=0,
        output_wav_path=None,
    ):
        if prompt_wav_path is None:
            prompt_waveform, prompt_text, spk_emb = None, None, None
            if use_zero_spk_emb:
                spk_emb = [
                    torch.zeros(1, 192, device=self.device, dtype=torch.bfloat16)
                ]
        else:
            paths = (
                prompt_wav_path
                if isinstance(prompt_wav_path, list)
                else [prompt_wav_path]
            )
            processed_prompts = [
                self.preprocess_one_prompt_wav(p, use_spk_emb) for p in paths
            ]
            waveforms_list, spk_emb = zip(*processed_prompts)
            prompt_waveform = torch.cat(waveforms_list, dim=-1)
            prompt_waveform = self.pad_waveform(prompt_waveform)
            spk_emb = list(spk_emb)
            if all([x is None for x in spk_emb]):
                spk_emb = None

        if instruction is not None:
            import json

            instruction = self.create_instruction(instruction)
            instruction = json.dumps(instruction, ensure_ascii=False)

        waveforms = []
        for text in text_list:
            spk_emb_copy = (
                [se.clone() for se in spk_emb] if spk_emb is not None else None
            )
            waveform = self.model.generate(
                prompt=prompt,
                text=text,
                spk_emb=spk_emb_copy,
                instruction=instruction,
                prompt_waveform=prompt_waveform,
                prompt_text=prompt_text,
                max_decode_steps=max_decode_steps,
                cfg=cfg,
                sigma=sigma,
                temperature=temperature,
                use_zero_spk_emb=use_zero_spk_emb,
            )
            waveforms.append(waveform)

        final_waveform = torch.cat(waveforms, dim=-1)

        if output_wav_path is not None:
            output_dir = os.path.dirname(output_wav_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            torchaudio.save(
                output_wav_path, final_waveform, sample_rate=self.sample_rate
            )
        return final_waveform


model = None
OUTPUT_DIR = "./output"
CONFIG_DIR = "./saved_configs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)


def load_model_fn(model_path):
    global model
    if model is None:
        logger.info(f"Loading model from {model_path}...")
        model = MingAudio(model_path)
        logger.info("Model loaded successfully!")
    return model


def copy_audio_to_config_dir(audio_path, config_name):
    if audio_path is None:
        return None
    config_audio_dir = os.path.join(CONFIG_DIR, config_name, "audio")
    os.makedirs(config_audio_dir, exist_ok=True)
    audio_ext = os.path.splitext(audio_path)[1]
    dest_path = os.path.join(config_audio_dir, f"ref_audio{audio_ext}")
    import shutil

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
        import shutil

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

    import json

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

    import json

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    return config_data, "配置加载成功"


def delete_config(config_name):
    if not config_name:
        return "请选择要删除的配置", False

    config_path = os.path.join(CONFIG_DIR, config_name)
    if not os.path.exists(config_path):
        return f"配置 '{config_name}' 不存在", False

    import shutil

    shutil.rmtree(config_path)
    return f"配置 '{config_name}' 已删除", True


def generate_speech(
    text,
    prompt_audio,
    prompt_text,
    emotion,
    dialect,
    style,
    speech_speed,
    pitch,
    volume,
    max_decode_steps,
    cfg,
    sigma,
    temperature,
    task_type,
    voice_description,
):
    global model, OUTPUT_DIR

    if not prompt_text:
        prompt_text = None

    if model is None:
        return None, "请先加载模型"

    if not text.strip():
        return None, "请输入文本内容"

    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text_list = [t.strip() for t in text.split("\n") if t.strip()]
    if not text_list:
        return None, "请输入文本内容"

    output_path = os.path.join(OUTPUT_DIR, "output.wav")

    use_spk_emb = prompt_audio is not None
    use_zero_spk_emb = prompt_audio is None

    instruction = {}
    # 音色描述优先：如果用户输入了音色描述，其他控制参数作为补充
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

    if task_type == "语音合成 (TTS)":
        prompt = "Please generate speech based on the following description.\n"
    elif task_type == "声音事件 (TTA)":
        prompt = "Please generate audio events based on given text.\n"
    elif task_type == "背景音乐 (BGM)":
        prompt = "Please generate music based on the following description.\n"
    else:
        prompt = "Please generate speech based on the following description.\n"

    try:
        if len(text_list) == 1:
            waveform = model.speech_generation(
                prompt=prompt,
                text=text_list[0],
                use_spk_emb=use_spk_emb,
                use_zero_spk_emb=use_zero_spk_emb,
                instruction=instruction if instruction else None,
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
                instruction=instruction if instruction else None,
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
        logger.error(f"Generation failed: {e}")
        return None, f"生成失败: {str(e)}"


def create_webui(
    model_path="./models/Ming-omni-tts-0.5B", load_model=True, external_model=None
):
    global model
    if load_model:
        load_model_fn(model_path)
    elif external_model is not None:
        model = external_model
    else:
        model = None
        logger.info("Running in demo mode without model")

    with gr.Blocks(title="Ming-Omni-TTS", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎤 Ming-Omni-TTS 语音合成")
        gr.Markdown("基于统一音频生成模型的文本转语音系统")

        with gr.Tab("语音合成 (TTS)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 参数配置")

                    input_text = gr.Textbox(
                        label="输入文本",
                        placeholder="请输入要合成语音的文本...",
                        lines=5,
                    )

                    prompt_audio = gr.Audio(
                        label="参考音频 (可选)",
                        type="filepath",
                    )

                    prompt_text = gr.Textbox(
                        label="参考文本 (可选)",
                        placeholder="参考音频对应的文本...",
                        lines=2,
                    )

                    with gr.Group():
                        gr.Markdown("#### 高级参数")
                        with gr.Row():
                            speech_speed = gr.Slider(
                                minimum=0.5,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                label="语速",
                            )
                            pitch = gr.Slider(
                                minimum=0.5,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                label="音高",
                            )
                            volume = gr.Slider(
                                minimum=0.5,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                label="音量",
                            )

                        max_decode_steps = gr.Slider(
                            minimum=100,
                            maximum=500,
                            value=200,
                            step=50,
                            label="解码步数",
                        )
                        cfg = gr.Slider(
                            minimum=1.0,
                            maximum=8.0,
                            value=2.0,
                            step=0.5,
                            label="CFG 强度",
                        )
                        sigma = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.25,
                            step=0.05,
                            label="Sigma",
                        )
                        temperature = gr.Slider(
                            minimum=0.0,
                            maximum=3.0,
                            value=0.0,
                            step=0.1,
                            label="Temperature",
                        )

                    with gr.Group():
                        gr.Markdown("#### 配置管理")
                        with gr.Row():
                            config_name_input = gr.Textbox(
                                label="配置名称",
                                placeholder="输入配置名称保存...",
                                scale=2,
                            )
                            save_config_btn = gr.Button("💾 保存配置", scale=1)
                        config_save_status = gr.Textbox(
                            label="保存状态",
                            interactive=False,
                            lines=1,
                        )

                        with gr.Row():
                            config_dropdown = gr.Dropdown(
                                choices=get_config_list(),
                                label="已保存配置",
                                scale=2,
                            )
                            refresh_configs_btn = gr.Button("🔄 刷新", scale=1)
                        with gr.Row():
                            load_config_btn = gr.Button("📂 加载配置", scale=1)
                            delete_config_btn = gr.Button(
                                "🗑️ 删除配置", variant="stop", scale=1
                            )
                        config_load_status = gr.Textbox(
                            label="加载状态",
                            interactive=False,
                            lines=1,
                        )

                with gr.Column(scale=1):
                    gr.Markdown("### 声音控制")
                    emotion = gr.Dropdown(
                        choices=[
                            "无",
                            "高兴",
                            "悲伤",
                            "愤怒",
                            "惊讶",
                            "恐惧",
                            "厌恶",
                        ],
                        label="情感",
                        value="无",
                    )
                    dialect = gr.Dropdown(
                        choices=[
                            "无",
                            "普通话",
                            "广粤话",
                            "四川话",
                            "东北话",
                            "河南话",
                        ],
                        label="方言",
                        value="无",
                    )
                    style = gr.Dropdown(
                        choices=[
                            "无",
                            "正式",
                            " casual",
                            "新闻播报",
                            "讲故事",
                            "ASMR耳语",
                        ],
                        label="风格",
                        value="无",
                    )
                    voice_description = gr.Textbox(
                        label="音色描述 (可选)",
                        placeholder="例如: 这是一位温柔的母亲声音，音色低沉浑厚，充满关爱",
                        lines=3,
                    )
                    with gr.Accordion("音色描述示例", open=False):
                        gr.Examples(
                            examples=[
                                ["一位温柔的母亲声音，音色低沉浑厚，充满关爱"],
                                ["年轻的男性主播，声音清澈明亮，富有活力"],
                                ["成熟的男性嗓音，声线低沉，带有一点沙哑"],
                                ["可爱的小女孩声音，甜美清脆，充满元气"],
                                ["一位威严的皇后，声音沉稳大气，充满威压"],
                                ["ASMR耳语，气音重，音量极低，语速极慢"],
                            ],
                            inputs=voice_description,
                        )

                    generate_btn = gr.Button("🎵 生成语音", variant="primary")

                    gr.Markdown("### 生成结果")
                    output_audio = gr.Audio(
                        label="生成的音频",
                        type="filepath",
                    )
                    status_msg = gr.Textbox(
                        label="状态",
                        interactive=False,
                    )
                    download_btn = gr.DownloadButton(
                        label="⬇️ 下载音频",
                        variant="secondary",
                    )

            generate_btn.click(
                fn=generate_speech,
                inputs=[
                    input_text,
                    prompt_audio,
                    prompt_text,
                    emotion,
                    dialect,
                    style,
                    speech_speed,
                    pitch,
                    volume,
                    max_decode_steps,
                    cfg,
                    sigma,
                    temperature,
                    gr.State("语音合成 (TTS)"),
                    voice_description,
                ],
                outputs=[output_audio, status_msg],
            )

            output_audio.change(
                fn=lambda x: x,
                inputs=output_audio,
                outputs=download_btn,
            )

            def on_save_config(
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
                msg, success = save_config(
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
                )
                if success:
                    new_choices = get_config_list()
                    return msg, gr.update(choices=new_choices), ""
                return msg, gr.update(), ""

            save_config_btn.click(
                fn=on_save_config,
                inputs=[
                    config_name_input,
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
                ],
                outputs=[config_save_status, config_dropdown, config_name_input],
            )

            def on_load_config(config_name):
                if not config_name:
                    return (
                        "请选择要加载的配置",
                        None,
                        "",
                        "无",
                        "无",
                        "无",
                        "",
                        1.0,
                        1.0,
                        1.0,
                        200,
                        2.0,
                        0.25,
                        0.0,
                    )
                config_data, msg = load_config(config_name)
                if config_data is None:
                    return (
                        msg,
                        None,
                        "",
                        "无",
                        "无",
                        "无",
                        "",
                        1.0,
                        1.0,
                        1.0,
                        200,
                        2.0,
                        0.25,
                        0.0,
                    )
                return (
                    msg,
                    config_data.get("prompt_audio"),
                    config_data.get("prompt_text", ""),
                    config_data.get("emotion", "无"),
                    config_data.get("dialect", "无"),
                    config_data.get("style", "无"),
                    config_data.get("voice_description", ""),
                    config_data.get("speech_speed", 1.0),
                    config_data.get("pitch", 1.0),
                    config_data.get("volume", 1.0),
                    config_data.get("max_decode_steps", 200),
                    config_data.get("cfg", 2.0),
                    config_data.get("sigma", 0.25),
                    config_data.get("temperature", 0.0),
                )

            load_config_btn.click(
                fn=on_load_config,
                inputs=config_dropdown,
                outputs=[
                    config_load_status,
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
                ],
            )

            refresh_configs_btn.click(
                fn=lambda: gr.update(choices=get_config_list()),
                outputs=config_dropdown,
            )

            def on_delete_config(config_name):
                if not config_name:
                    return "请选择要删除的配置", gr.update(choices=get_config_list())
                msg, success = delete_config(config_name)
                if success:
                    return msg, gr.update(choices=get_config_list(), value=None)
                return msg, gr.update()

            delete_config_btn.click(
                fn=on_delete_config,
                inputs=config_dropdown,
                outputs=[config_load_status, config_dropdown],
            )

        with gr.Tab("声音事件 (TTA)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 声音事件描述")
                    input_text_tta = gr.Textbox(
                        label="输入描述",
                        placeholder="例如: Thunder and a gentle rain",
                        lines=3,
                    )

                    max_decode_steps_tta = gr.Slider(
                        minimum=100,
                        maximum=500,
                        value=200,
                        step=50,
                        label="解码步数",
                    )
                    cfg_tta = gr.Slider(
                        minimum=1.0,
                        maximum=8.0,
                        value=4.5,
                        step=0.5,
                        label="CFG 强度",
                    )
                    sigma_tta = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.3,
                        step=0.05,
                        label="Sigma",
                    )
                    temperature_tta = gr.Slider(
                        minimum=0.0,
                        maximum=3.0,
                        value=2.5,
                        step=0.1,
                        label="Temperature",
                    )

                    generate_btn_tta = gr.Button("🎵 生成声音", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("### 生成结果")
                    output_audio_tta = gr.Audio(
                        label="生成的音频",
                        type="filepath",
                    )
                    status_msg_tta = gr.Textbox(
                        label="状态",
                        interactive=False,
                    )

            generate_btn_tta.click(
                fn=generate_speech,
                inputs=[
                    input_text_tta,
                    gr.State(None),
                    gr.State(None),
                    gr.State("无"),
                    gr.State("无"),
                    gr.State("无"),
                    gr.State(1.0),
                    gr.State(1.0),
                    gr.State(1.0),
                    max_decode_steps_tta,
                    cfg_tta,
                    sigma_tta,
                    temperature_tta,
                    gr.State("声音事件 (TTA)"),
                    gr.State(None),
                ],
                outputs=[output_audio_tta, status_msg_tta],
            )

        with gr.Tab("背景音乐 (BGM)"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 音乐描述")

                    genre = gr.Dropdown(
                        choices=[
                            "无",
                            "电子舞曲",
                            "古典",
                            "流行",
                            "摇滚",
                            "爵士",
                            "乡村",
                            "嘻哈",
                        ],
                        label="音乐风格",
                        value="无",
                    )
                    mood = gr.Dropdown(
                        choices=["无", "欢快", "悲伤", "紧张", "放松", "神秘", "浪漫"],
                        label="情绪",
                        value="无",
                    )
                    instrument = gr.Dropdown(
                        choices=["无", "钢琴", "吉他", "架子鼓", "小提琴", "电吉他"],
                        label="主乐器",
                        value="无",
                    )
                    theme = gr.Dropdown(
                        choices=["无", "节日", "运动", "放松", "工作", "电影"],
                        label="主题",
                        value="无",
                    )

                    max_decode_steps_bgm = gr.Slider(
                        minimum=200,
                        maximum=600,
                        value=400,
                        step=50,
                        label="解码步数",
                    )

                    generate_btn_bgm = gr.Button("🎵 生成音乐", variant="primary")

                with gr.Column(scale=1):
                    gr.Markdown("### 生成结果")
                    output_audio_bgm = gr.Audio(
                        label="生成的音频",
                        type="filepath",
                    )
                    status_msg_bgm = gr.Textbox(
                        label="状态",
                        interactive=False,
                    )

            def generate_bgm(genre, mood, instrument, theme, max_decode_steps):
                text_parts = []
                if genre != "无":
                    text_parts.append(f"Genre: {genre}")
                if mood != "无":
                    text_parts.append(f"Mood: {mood}")
                if instrument != "无":
                    text_parts.append(f"Instrument: {instrument}")
                if theme != "无":
                    text_parts.append(f"Theme: {theme}")

                text = " " + " ".join(text_parts) if text_parts else " "

                return generate_speech(
                    text,
                    None,
                    None,
                    "无",
                    "无",
                    "无",
                    1.0,
                    1.0,
                    1.0,
                    max_decode_steps,
                    2.0,
                    0.25,
                    0.0,
                    "背景音乐 (BGM)",
                    None,
                )

            generate_btn_bgm.click(
                fn=generate_bgm,
                inputs=[genre, mood, instrument, theme, max_decode_steps_bgm],
                outputs=[output_audio_bgm, status_msg_bgm],
            )

    return demo


def create_api(model):
    app = Flask(__name__)
    CORS(app)

    @app.route("/")
    def handle_request():
        text = request.args.get("text", "")
        speaker = request.args.get("speaker", "京京")

        if not text:
            return """<!DOCTYPE html>
<html>
<head>
    <title>Ming-Omni-TTS WebUI</title>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; background: #f5f5f5; }
        h1 { color: #333; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input[type="text"], textarea { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
        button { background: #4CAF50; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover { background: #45a049; }
        #result { margin-top: 20px; }
        audio { width: 100%; margin-top: 10px; }
        .info { background: #e3f2fd; padding: 15px; border-radius: 4px; margin-bottom: 20px; border-left: 4px solid #2196F3; }
        .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Ming-Omni-TTS 语音合成</h1>
        <div class="info">
            <p><strong>📡 API 调用方式：</strong></p>
            <code>GET http://your-server:7860/?text=要合成的文本&speaker=配置文件名</code>
            <p>例如：<code>http://your-server:7860/?text=你好世界&speaker=京京</code></p>
            <p><strong>📂 可用配置文件：</strong> 京京, 广末凉子, 苏瑶, 走心女主播</p>
        </div>
        <div class="form-group">
            <label>输入文本：</label>
            <textarea id="text" rows="3" placeholder="请输入要合成语音的文本..."></textarea>
        </div>
        <div class="form-group">
            <label>说话人配置：</label>
            <input type="text" id="speaker" value="京京" placeholder="配置文件名，如：京京">
        </div>
        <button onclick="generate()">🎵 生成语音</button>
        <div id="result"></div>
    </div>
    <script>
        async function generate() {
            const text = document.getElementById('text').value;
            const speaker = document.getElementById('speaker').value;
            const result = document.getElementById('result');
            
            if (!text) {
                result.innerHTML = '<p style="color:red;">请输入文本</p>';
                return;
            }
            
            result.innerHTML = '<p>⏳ 正在生成...</p>';
            
            try {
                const url = '/?text=' + encodeURIComponent(text) + '&speaker=' + encodeURIComponent(speaker);
                const response = await fetch(url);
                
                if (!response.ok) {
                    const error = await response.text();
                    result.innerHTML = '<p style="color:red;">❌ 错误: ' + error + '</p>';
                    return;
                }
                
                const blob = await response.blob();
                const audioUrl = URL.createObjectURL(blob);
                result.innerHTML = '<audio controls src="' + audioUrl + '"></audio>';
            } catch (e) {
                result.innerHTML = '<p style="color:red;">❌ 错误: ' + e.message + '</p>';
            }
        }
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

        instruction = {}
        if config_data.get("emotion"):
            instruction["情感"] = config_data["emotion"]
        if config_data.get("dialect"):
            instruction["方言"] = config_data["dialect"]
        if config_data.get("style"):
            instruction["风格"] = config_data["style"]

        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        text_list = [t.strip() for t in text.split("\n") if t.strip()]

        output_path = os.path.join(OUTPUT_DIR, f"api_{uuid.uuid4().hex}.wav")

        try:
            if len(text_list) == 1:
                waveform = model.speech_generation(
                    prompt="Please generate speech based on the following description.\n",
                    text=text_list[0],
                    use_spk_emb=config_data.get("prompt_audio") is not None,
                    use_zero_spk_emb=config_data.get("prompt_audio") is None,
                    instruction=instruction if instruction else None,
                    prompt_wav_path=config_data.get("prompt_audio"),
                    prompt_text=config_data.get("prompt_text"),
                    max_decode_steps=config_data.get("max_decode_steps", 200),
                    cfg=config_data.get("cfg", 2.0),
                    sigma=config_data.get("sigma", 0.25),
                    temperature=config_data.get("temperature", 0.0),
                    output_wav_path=output_path,
                )
            else:
                waveform = model.speech_generation_batch(
                    prompt="Please generate speech based on the following description.\n",
                    text_list=text_list,
                    use_spk_emb=config_data.get("prompt_audio") is not None,
                    use_zero_spk_emb=config_data.get("prompt_audio") is None,
                    instruction=instruction if instruction else None,
                    prompt_wav_path=config_data.get("prompt_audio"),
                    prompt_text=config_data.get("prompt_text"),
                    max_decode_steps=config_data.get("max_decode_steps", 200),
                    cfg=config_data.get("cfg", 2.0),
                    sigma=config_data.get("sigma", 0.25),
                    temperature=config_data.get("temperature", 0.0),
                    output_wav_path=output_path,
                )
            logger.info(f"生成成功: {output_path} (共 {len(text_list)} 段)")
            return send_file(output_path, mimetype="audio/wav", as_attachment=True)
        except Exception as e:
            logger.error(f"生成失败: {e}")
            return f"生成失败: {str(e)}", 500

    return app


if __name__ == "__main__":
    model_path = os.environ.get("MODEL_PATH", "./models/Ming-omni-tts-0.5B")
    port = int(os.environ.get("PORT", 7860))
    load_model = os.environ.get("LOAD_MODEL", "true").lower() == "true"

    if load_model:
        load_model_fn(model_path)

    demo = create_webui(model_path, load_model=False, external_model=model)

    from werkzeug.serving import run_simple
    import threading

    def run_flask():
        flask_app = create_api(model)
        run_simple("0.0.0.0", port, flask_app, use_reloader=False, use_debugger=False)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    gradio_port = port + 1

    print(f"\n{'=' * 60}")
    print(f"🎤 Ming-Omni-TTS 服务已启动!")
    print(f"{'=' * 60}")
    print(f"📡 API 服务:   http://localhost:{port}/")
    print(f"   调用示例: http://localhost:{port}/?text=你好世界&speaker=京京")
    print(f"🎨 Gradio:    http://localhost:{gradio_port}/")
    print(f"{'=' * 60}\n")

    demo.launch(server_name="0.0.0.0", server_port=gradio_port, share=False)
