#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

import os
import sys
import uuid
import json
import warnings
import torch
import torchaudio
from transformers import AutoTokenizer
from flask import Flask, request, send_file, jsonify, render_template_string
from flask_cors import CORS
from loguru import logger

warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from modeling_bailingmm import BailingMMNativeForConditionalGeneration
from sentence_manager.sentence_manager import SentenceNormalizer
from spkemb_extractor import SpkembExtractor

OUTPUT_DIR = "./output"
CONFIG_DIR = "./saved_configs"
UPLOAD_DIR = "./uploads"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


def copy_audio_to_config_dir(audio_path, config_name):
    if audio_path is None:
        return None
    import shutil

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
    import shutil

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
    import shutil

    if not config_name:
        return "请选择要删除的配置", False

    config_path = os.path.join(CONFIG_DIR, config_name)
    if not os.path.exists(config_path):
        return f"配置 '{config_name}' 不存在", False

    shutil.rmtree(config_path)
    return f"配置 '{config_name}' 已删除", True


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


def load_model_fn(model_path):
    global model
    if model is None:
        logger.info(f"Loading model from {model_path}...")
        model = MingAudio(model_path)
        logger.info("Model loaded successfully!")
    return model


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

    app = Flask(__name__)
    CORS(app)

    @app.route("/")
    def index():
        config_list = get_config_list()
        return render_template_string(HTML_TEMPLATE, config_list=config_list)

    @app.route("/configs", methods=["GET"])
    def list_configs():
        return jsonify(get_config_list())

    @app.route("/save_config", methods=["POST"])
    def api_save_config():
        data = request.form or request.json
        msg, success = save_config(
            config_name=data.get("config_name"),
            prompt_audio=data.get("prompt_audio"),
            prompt_text=data.get("prompt_text"),
            emotion=data.get("emotion"),
            dialect=data.get("dialect"),
            style=data.get("style"),
            voice_description=data.get("voice_description"),
            speech_speed=float(data.get("speech_speed", 1.0)),
            pitch=float(data.get("pitch", 1.0)),
            volume=float(data.get("volume", 1.0)),
            max_decode_steps=int(data.get("max_decode_steps", 200)),
            cfg=float(data.get("cfg", 2.0)),
            sigma=float(data.get("sigma", 0.25)),
            temperature=float(data.get("temperature", 0.0)),
        )
        return jsonify({"success": success, "message": msg})

    @app.route("/load_config", methods=["GET"])
    def api_load_config():
        config_name = request.args.get("config_name")
        config_data, msg = load_config(config_name)
        if config_data is None:
            return jsonify({"success": False, "message": msg}), 400
        return jsonify({"success": True, "message": msg, "data": config_data})

    @app.route("/delete_config", methods=["POST"])
    def api_delete_config():
        data = request.form or request.json
        config_name = data.get("config_name")
        msg, success = delete_config(config_name)
        return jsonify({"success": success, "message": msg})

    @app.route("/upload", methods=["POST"])
    def upload_audio():
        if "file" not in request.files:
            return jsonify({"success": False, "message": "没有文件"}), 400
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "message": "文件名为空"}), 400
        ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(UPLOAD_DIR, filename)
        file.save(filepath)
        return jsonify({"success": True, "filepath": filepath})

    @app.route("/generate", methods=["POST"])
    def generate():
        if model is None:
            return jsonify({"success": False, "message": "模型未加载"}), 400

        data = request.form or request.json
        task_type = data.get("task_type", "语音合成 (TTS)")
        text = data.get("text", "")
        prompt_audio = data.get("prompt_audio")
        prompt_text = data.get("prompt_text")
        emotion = data.get("emotion", "无")
        dialect = data.get("dialect", "无")
        style = data.get("style", "无")
        speech_speed = float(data.get("speech_speed", 1.0))
        pitch = float(data.get("pitch", 1.0))
        volume = float(data.get("volume", 1.0))
        max_decode_steps = int(data.get("max_decode_steps", 200))
        cfg = float(data.get("cfg", 2.0))
        sigma = float(data.get("sigma", 0.25))
        temperature = float(data.get("temperature", 0.0))
        voice_description = data.get("voice_description")

        from inference import generate_speech as _generate_speech

        output_filename = f"webui_{uuid.uuid4().hex}.wav"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        result = _generate_speech(
            model=model,
            text=text,
            task_type=task_type,
            prompt_audio=prompt_audio,
            prompt_text=prompt_text,
            emotion=emotion,
            dialect=dialect,
            style=style,
            speech_speed=speech_speed,
            pitch=pitch,
            volume=volume,
            max_decode_steps=max_decode_steps,
            cfg=cfg,
            sigma=sigma,
            temperature=temperature,
            voice_description=voice_description,
            output_path=output_path,
        )

        if result[0] is None:
            return jsonify({"success": False, "message": result[1]}), 500

        return jsonify(
            {
                "success": True,
                "message": result[1],
                "audio_url": f"/audio/{output_filename}",
            }
        )

    @app.route("/audio/<filename>")
    def serve_audio(filename):
        filepath = os.path.join(OUTPUT_DIR, filename)
        if not os.path.exists(filepath):
            return "File not found", 404
        return send_file(filepath, mimetype="audio/wav")

    @app.route("/config_audio/<config_name>")
    def serve_config_audio(config_name):
        config_path = os.path.join(CONFIG_DIR, config_name)
        if not os.path.exists(config_path):
            return "Config not found", 404
        for fname in os.listdir(os.path.join(config_path, "audio")):
            filepath = os.path.join(config_path, "audio", fname)
            if os.path.isfile(filepath):
                ext = os.path.splitext(fname)[1]
                mimetype = (
                    f"audio/{ext[1:]}"
                    if ext[1:] in ["wav", "mp3", "ogg"]
                    else "audio/wav"
                )
                return send_file(filepath, mimetype=mimetype)
        return "Audio not found", 404

    return app


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Ming-Omni-TTS WebUI</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 20px; }
        h1 { color: #333; text-align: center; }
        .tabs { display: flex; border-bottom: 2px solid #ddd; margin-bottom: 20px; }
        .tab { padding: 12px 24px; cursor: pointer; border: none; background: none; font-size: 16px; color: #666; }
        .tab.active { border-bottom: 2px solid #4CAF50; color: #4CAF50; font-weight: bold; }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .row { display: flex; gap: 20px; }
        .col { flex: 1; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input[type="text"], textarea, select { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; font-size: 14px; }
        textarea { resize: vertical; }
        input[type="range"] { width: 100%; }
        .slider-group { display: flex; align-items: center; gap: 10px; }
        .slider-group input[type="range"] { flex: 1; }
        .slider-group span { min-width: 40px; text-align: right; }
        button { background: #4CAF50; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        button:hover { background: #45a049; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        button.secondary { background: #2196F3; }
        button.danger { background: #f44336; }
        .result { margin-top: 20px; padding: 15px; border-radius: 4px; }
        .result.success { background: #e8f5e9; border: 1px solid #4CAF50; }
        .result.error { background: #ffebee; border: 1px solid #f44336; }
        audio { width: 100%; margin-top: 10px; }
        .config-section { background: #f9f9f9; padding: 15px; border-radius: 4px; margin-top: 20px; }
        .examples { font-size: 12px; color: #666; }
        .examples span { display: block; padding: 3px 0; cursor: pointer; }
        .examples span:hover { color: #4CAF50; }
        .drop-zone { border: 2px dashed #ddd; border-radius: 4px; padding: 20px; text-align: center; cursor: pointer; transition: all 0.3s; }
        .drop-zone:hover, .drop-zone.dragover { border-color: #4CAF50; background: #f9f9f9; }
        .drop-zone input { display: none; }
        .drop-zone p { margin: 0; color: #666; font-size: 14px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Ming-Omni-TTS 语音合成</h1>
        
        <div class="tabs">
            <button class="tab active" onclick="switchTab('tts')">语音合成 (TTS)</button>
            <button class="tab" onclick="switchTab('tta')">声音事件 (TTA)</button>
            <button class="tab" onclick="switchTab('bgm')">背景音乐 (BGM)</button>
        </div>

        <!-- TTS Tab -->
        <div id="tts" class="tab-content active">
            <div class="row">
                <div class="col">
                    <div class="form-group">
                        <label>输入文本：</label>
                        <textarea id="tts_text" rows="4" placeholder="请输入要合成语音的文本..."></textarea>
                    </div>
                    <div class="form-group">
                        <label>参考音频 (可选)：</label>
                        <div class="drop-zone" id="tts_drop_zone">
                            <input type="file" id="tts_prompt_audio" accept="audio/*">
                            <p>拖拽音频文件到此处 或 点击选择</p>
                        </div>
                        <div id="tts_prompt_audio_display"></div>
                    </div>
                    <div class="form-group">
                        <label>参考文本 (可选)：</label>
                        <textarea id="tts_prompt_text" rows="2" placeholder="参考音频对应的文本..."></textarea>
                    </div>
                    
                    <div class="form-group">
                        <label>语速：</label>
                        <div class="slider-group">
                            <input type="range" id="tts_speech_speed" min="0.5" max="2.0" step="0.1" value="1.0">
                            <span id="tts_speech_speed_val">1.0</span>
                        </div>
                    </div>
                    <div class="form-group">
                        <label>音高：</label>
                        <div class="slider-group">
                            <input type="range" id="tts_pitch" min="0.5" max="2.0" step="0.1" value="1.0">
                            <span id="tts_pitch_val">1.0</span>
                        </div>
                    </div>
                    <div class="form-group">
                        <label>音量：</label>
                        <div class="slider-group">
                            <input type="range" id="tts_volume" min="0.5" max="2.0" step="0.1" value="1.0">
                            <span id="tts_volume_val">1.0</span>
                        </div>
                    </div>
                    <div class="form-group">
                        <label>解码步数：</label>
                        <div class="slider-group">
                            <input type="range" id="tts_max_decode_steps" min="100" max="500" step="50" value="200">
                            <span id="tts_max_decode_steps_val">200</span>
                        </div>
                    </div>
                    <div class="form-group">
                        <label>CFG 强度：</label>
                        <div class="slider-group">
                            <input type="range" id="tts_cfg" min="1.0" max="8.0" step="0.5" value="2.0">
                            <span id="tts_cfg_val">2.0</span>
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Sigma：</label>
                        <div class="slider-group">
                            <input type="range" id="tts_sigma" min="0.1" max="1.0" step="0.05" value="0.25">
                            <span id="tts_sigma_val">0.25</span>
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Temperature：</label>
                        <div class="slider-group">
                            <input type="range" id="tts_temperature" min="0.0" max="3.0" step="0.1" value="0.0">
                            <span id="tts_temperature_val">0.0</span>
                        </div>
                    </div>
                </div>
                
                <div class="col">
                    <div class="form-group">
                        <label>情感：</label>
                        <select id="tts_emotion">
                            <option value="无">无</option>
                            <option value="高兴">高兴</option>
                            <option value="悲伤">悲伤</option>
                            <option value="愤怒">愤怒</option>
                            <option value="惊讶">惊讶</option>
                            <option value="恐惧">恐惧</option>
                            <option value="厌恶">厌恶</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>方言：</label>
                        <select id="tts_dialect">
                            <option value="无">无</option>
                            <option value="普通话">普通话</option>
                            <option value="广粤话">广粤话</option>
                            <option value="四川话">四川话</option>
                            <option value="东北话">东北话</option>
                            <option value="河南话">河南话</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>风格：</label>
                        <select id="tts_style">
                            <option value="无">无</option>
                            <option value="正式">正式</option>
                            <option value="casual">casual</option>
                            <option value="新闻播报">新闻播报</option>
                            <option value="讲故事">讲故事</option>
                            <option value="ASMR耳语">ASMR耳语</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>音色描述 (可选)：</label>
                        <textarea id="tts_voice_description" rows="3" placeholder="例如: 这是一位温柔的母亲声音，音色低沉浑厚，充满关爱"></textarea>
                        <div class="examples">
                            <span onclick="setVoiceDesc('一位温柔的母亲声音，音色低沉浑厚，充满关爱')">一位温柔的母亲声音</span>
                            <span onclick="setVoiceDesc('年轻的男性主播，声音清澈明亮，富有活力')">年轻的男性主播</span>
                            <span onclick="setVoiceDesc('成熟的男性嗓音，声线低沉，带有一点沙哑')">成熟的男性嗓音</span>
                            <span onclick="setVoiceDesc('可爱的小女孩声音，甜美清脆，充满元气')">可爱的小女孩声音</span>
                            <span onclick="setVoiceDesc('ASMR耳语，气音重，音量极低，语速极慢')">ASMR耳语</span>
                        </div>
                    </div>
                    
                    <button id="tts_generate" onclick="generateTTS()">生成语音</button>
                    
                    <div id="tts_result"></div>
                </div>
            </div>
            
            <div class="config-section">
                <h3>配置管理</h3>
                <div class="row">
                    <div class="col">
                        <div class="form-group">
                            <label>配置名称：</label>
                            <input type="text" id="config_name" placeholder="输入配置名称保存...">
                        </div>
                        <button onclick="saveConfig()">保存配置</button>
                    </div>
                    <div class="col">
                        <div class="form-group">
                            <label>已保存配置：</label>
                            <select id="config_list">
                                <option value="">加载配置...</option>
                            </select>
                        </div>
                        <button class="secondary" onclick="loadConfig()">加载配置</button>
                        <button class="danger" onclick="deleteConfig()">删除配置</button>
                    </div>
                </div>
                <div id="config_msg"></div>
            </div>
        </div>

        <!-- TTA Tab -->
        <div id="tta" class="tab-content">
            <div class="row">
                <div class="col">
                    <div class="form-group">
                        <label>声音事件描述：</label>
                        <textarea id="tta_text" rows="4" placeholder="例如: Thunder and a gentle rain"></textarea>
                    </div>
                    <div class="form-group">
                        <label>解码步数：</label>
                        <div class="slider-group">
                            <input type="range" id="tta_max_decode_steps" min="100" max="500" step="50" value="200">
                            <span id="tta_max_decode_steps_val">200</span>
                        </div>
                    </div>
                    <div class="form-group">
                        <label>CFG 强度：</label>
                        <div class="slider-group">
                            <input type="range" id="tta_cfg" min="1.0" max="8.0" step="0.5" value="4.5">
                            <span id="tta_cfg_val">4.5</span>
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Sigma：</label>
                        <div class="slider-group">
                            <input type="range" id="tta_sigma" min="0.1" max="1.0" step="0.05" value="0.3">
                            <span id="tta_sigma_val">0.3</span>
                        </div>
                    </div>
                    <div class="form-group">
                        <label>Temperature：</label>
                        <div class="slider-group">
                            <input type="range" id="tta_temperature" min="0.0" max="3.0" step="0.1" value="2.5">
                            <span id="tta_temperature_val">2.5</span>
                        </div>
                    </div>
                    <button onclick="generateTTA()">生成声音</button>
                </div>
                <div class="col">
                    <div id="tta_result"></div>
                </div>
            </div>
        </div>

        <!-- BGM Tab -->
        <div id="bgm" class="tab-content">
            <div class="row">
                <div class="col">
                    <div class="form-group">
                        <label>音乐风格：</label>
                        <select id="bgm_genre">
                            <option value="无">无</option>
                            <option value="电子舞曲">电子舞曲</option>
                            <option value="古典">古典</option>
                            <option value="流行">流行</option>
                            <option value="摇滚">摇滚</option>
                            <option value="爵士">爵士</option>
                            <option value="乡村">乡村</option>
                            <option value="嘻哈">嘻哈</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>情绪：</label>
                        <select id="bgm_mood">
                            <option value="无">无</option>
                            <option value="欢快">欢快</option>
                            <option value="悲伤">悲伤</option>
                            <option value="紧张">紧张</option>
                            <option value="放松">放松</option>
                            <option value="神秘">神秘</option>
                            <option value="浪漫">浪漫</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>主乐器：</label>
                        <select id="bgm_instrument">
                            <option value="无">无</option>
                            <option value="钢琴">钢琴</option>
                            <option value="吉他">吉他</option>
                            <option value="架子鼓">架子鼓</option>
                            <option value="小提琴">小提琴</option>
                            <option value="电吉他">电吉他</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>主题：</label>
                        <select id="bgm_theme">
                            <option value="无">无</option>
                            <option value="节日">节日</option>
                            <option value="运动">运动</option>
                            <option value="放松">放松</option>
                            <option value="工作">工作</option>
                            <option value="电影">电影</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>解码步数：</label>
                        <div class="slider-group">
                            <input type="range" id="bgm_max_decode_steps" min="200" max="600" step="50" value="400">
                            <span id="bgm_max_decode_steps_val">400</span>
                        </div>
                    </div>
                    <button onclick="generateBGM()">生成音乐</button>
                </div>
                <div class="col">
                    <div id="bgm_result"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Drop zone drag events
        var dropZone = document.getElementById('tts_drop_zone');
        var fileInput = document.getElementById('tts_prompt_audio');
        
        dropZone.addEventListener('click', function() {
            fileInput.click();
        });
        
        dropZone.addEventListener('dragover', function(e) {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        
        dropZone.addEventListener('dragleave', function() {
            dropZone.classList.remove('dragover');
        });
        
        dropZone.addEventListener('drop', function(e) {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            var files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                var event = new Event('change', { bubbles: true });
                fileInput.dispatchEvent(event);
            }
        });
        
        // Handle file input change
        fileInput.addEventListener('change', function() {
            if (this.files && this.files[0]) {
                var file = this.files[0];
                var audioDisplay = document.getElementById('tts_prompt_audio_display');
                var url = URL.createObjectURL(file);
                audioDisplay.innerHTML = '<audio controls src="' + url + '"></audio><p style="font-size:12px;color:#666;">' + file.name + '</p>';
            }
        });
        
        // Slider value display
        document.querySelectorAll('input[type="range"]').forEach(function(slider) {
            slider.addEventListener('input', function() {
                var span = document.getElementById(this.id + '_val');
                if (span) span.textContent = this.value;
            });
        });

        function switchTab(tabId) {
            document.querySelectorAll('.tab').forEach(function(t) { t.classList.remove('active'); });
            document.querySelectorAll('.tab-content').forEach(function(c) { c.classList.remove('active'); });
            document.querySelector('.tab[onclick="switchTab(\\'' + tabId + '\\')"]').classList.add('active');
            document.getElementById(tabId).classList.add('active');
        }

        function setVoiceDesc(text) {
            document.getElementById('tts_voice_description').value = text;
        }

        function showResult(id, success, message, audioUrl) {
            var resultDiv = document.getElementById(id + '_result');
            if (success) {
                resultDiv.className = 'result success';
                resultDiv.innerHTML = '<p>' + message + '</p><audio controls src="' + audioUrl + '"></audio>';
            } else {
                resultDiv.className = 'result error';
                resultDiv.innerHTML = '<p>' + message + '</p>';
            }
        }

        function getFormData(prefix) {
            return {
                text: document.getElementById(prefix + '_text').value,
                prompt_audio: document.getElementById(prefix + '_prompt_audio') ? document.getElementById(prefix + '_prompt_audio').dataset.filepath : null,
                prompt_text: document.getElementById(prefix + '_prompt_text') ? document.getElementById(prefix + '_prompt_text').value : null,
                emotion: document.getElementById(prefix + '_emotion') ? document.getElementById(prefix + '_emotion').value : '无',
                dialect: document.getElementById(prefix + '_dialect') ? document.getElementById(prefix + '_dialect').value : '无',
                style: document.getElementById(prefix + '_style') ? document.getElementById(prefix + '_style').value : '无',
                speech_speed: document.getElementById(prefix + '_speech_speed') ? parseFloat(document.getElementById(prefix + '_speech_speed').value) : 1.0,
                pitch: document.getElementById(prefix + '_pitch') ? parseFloat(document.getElementById(prefix + '_pitch').value) : 1.0,
                volume: document.getElementById(prefix + '_volume') ? parseFloat(document.getElementById(prefix + '_volume').value) : 1.0,
                max_decode_steps: document.getElementById(prefix + '_max_decode_steps') ? parseInt(document.getElementById(prefix + '_max_decode_steps').value) : 200,
                cfg: document.getElementById(prefix + '_cfg') ? parseFloat(document.getElementById(prefix + '_cfg').value) : 2.0,
                sigma: document.getElementById(prefix + '_sigma') ? parseFloat(document.getElementById(prefix + '_sigma').value) : 0.25,
                temperature: document.getElementById(prefix + '_temperature') ? parseFloat(document.getElementById(prefix + '_temperature').value) : 0.0,
                voice_description: document.getElementById(prefix + '_voice_description') ? document.getElementById(prefix + '_voice_description').value : null,
            };
        }

        async function uploadAudioIfNeeded(inputId) {
            var input = document.getElementById(inputId);
            if (input && input.files && input.files[0]) {
                var formData = new FormData();
                formData.append('file', input.files[0]);
                var resp = await fetch('/upload', { method: 'POST', body: formData });
                var data = await resp.json();
                if (data.success) {
                    input.dataset.filepath = data.filepath;
                    return data.filepath;
                }
            }
            return null;
        }

        async function generateTTS() {
            var btn = document.getElementById('tts_generate');
            btn.disabled = true;
            btn.textContent = '生成中...';

            var text = document.getElementById('tts_text').value;
            if (!text) {
                showResult('tts', false, '请输入文本');
                btn.disabled = false;
                btn.textContent = '生成语音';
                return;
            }

            await uploadAudioIfNeeded('tts_prompt_audio');

            var data = getFormData('tts');
            data.task_type = '语音合成 (TTS)';

            try {
                var resp = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                var result = await resp.json();
                if (result.success) {
                    showResult('tts', true, result.message, result.audio_url);
                } else {
                    showResult('tts', false, result.message);
                }
            } catch (e) {
                showResult('tts', false, '错误: ' + e.message);
            }

            btn.disabled = false;
            btn.textContent = '生成语音';
        }

        async function generateTTA() {
            var text = document.getElementById('tta_text').value;
            if (!text) {
                showResult('tta', false, '请输入描述');
                return;
            }

            var data = {
                task_type: '声音事件 (TTA)',
                text: text,
                max_decode_steps: parseInt(document.getElementById('tta_max_decode_steps').value),
                cfg: parseFloat(document.getElementById('tta_cfg').value),
                sigma: parseFloat(document.getElementById('tta_sigma').value),
                temperature: parseFloat(document.getElementById('tta_temperature').value),
            };

            try {
                var resp = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                var result = await resp.json();
                if (result.success) {
                    showResult('tta', true, result.message, result.audio_url);
                } else {
                    showResult('tta', false, result.message);
                }
            } catch (e) {
                showResult('tta', false, '错误: ' + e.message);
            }
        }

        async function generateBGM() {
            var genre = document.getElementById('bgm_genre').value;
            var mood = document.getElementById('bgm_mood').value;
            var instrument = document.getElementById('bgm_instrument').value;
            var theme = document.getElementById('bgm_theme').value;

            var textParts = [];
            if (genre !== '无') textParts.push('Genre: ' + genre);
            if (mood !== '无') textParts.push('Mood: ' + mood);
            if (instrument !== '无') textParts.push('Instrument: ' + instrument);
            if (theme !== '无') textParts.push('Theme: ' + theme);

            var text = textParts.length > 0 ? ' ' + textParts.join(' ') : ' ';

            var data = {
                task_type: '背景音乐 (BGM)',
                text: text,
                max_decode_steps: parseInt(document.getElementById('bgm_max_decode_steps').value),
                emotion: '无',
                dialect: '无',
                style: '无',
                speech_speed: 1.0,
                pitch: 1.0,
                volume: 1.0,
                cfg: 2.0,
                sigma: 0.25,
                temperature: 0.0,
            };

            try {
                var resp = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                var result = await resp.json();
                if (result.success) {
                    showResult('bgm', true, result.message, result.audio_url);
                } else {
                    showResult('bgm', false, result.message);
                }
            } catch (e) {
                showResult('bgm', false, '错误: ' + e.message);
            }
        }

        // Config management
        async function loadConfigList() {
            try {
                var resp = await fetch('/configs');
                var configs = await resp.json();
                var select = document.getElementById('config_list');
                select.innerHTML = '<option value="">加载配置...</option>';
                configs.forEach(function(c) {
                    var opt = document.createElement('option');
                    opt.value = c;
                    opt.textContent = c;
                    select.appendChild(opt);
                });
            } catch (e) {}
        }

        async function saveConfig() {
            var configName = document.getElementById('config_name').value;
            if (!configName) {
                document.getElementById('config_msg').textContent = '请输入配置名称';
                return;
            }

            await uploadAudioIfNeeded('tts_prompt_audio');

            var data = getFormData('tts');
            data.config_name = configName;

            try {
                var resp = await fetch('/save_config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });
                var result = await resp.json();
                document.getElementById('config_msg').textContent = result.message;
                loadConfigList();
            } catch (e) {
                document.getElementById('config_msg').textContent = '错误: ' + e.message;
            }
        }

        async function loadConfig() {
            var configName = document.getElementById('config_list').value;
            if (!configName) {
                document.getElementById('config_msg').textContent = '请选择配置';
                return;
            }

            try {
                var resp = await fetch('/load_config?config_name=' + encodeURIComponent(configName));
                var result = await resp.json();
                if (result.success) {
                    var data = result.data;
                    document.getElementById('tts_prompt_text').value = data.prompt_text || '';
                    
                    // 显示已保存的音频
                    var audioDisplay = document.getElementById('tts_prompt_audio_display');
                    if (data.prompt_audio) {
                        audioDisplay.innerHTML = '<audio controls src="/config_audio/' + encodeURIComponent(data.name) + '"></audio><p style="font-size:12px;color:#666;">已保存的参考音频</p>';
                        document.getElementById('tts_prompt_audio').dataset.filepath = data.prompt_audio;
                    } else {
                        audioDisplay.innerHTML = '';
                    }
                    document.getElementById('tts_emotion').value = data.emotion || '无';
                    document.getElementById('tts_dialect').value = data.dialect || '无';
                    document.getElementById('tts_style').value = data.style || '无';
                    document.getElementById('tts_voice_description').value = data.voice_description || '';
                    setSlider('tts_speech_speed', data.speech_speed || 1.0);
                    setSlider('tts_pitch', data.pitch || 1.0);
                    setSlider('tts_volume', data.volume || 1.0);
                    setSlider('tts_max_decode_steps', data.max_decode_steps || 200);
                    setSlider('tts_cfg', data.cfg || 2.0);
                    setSlider('tts_sigma', data.sigma || 0.25);
                    setSlider('tts_temperature', data.temperature || 0.0);
                    document.getElementById('config_msg').textContent = result.message;
                } else {
                    document.getElementById('config_msg').textContent = result.message;
                }
            } catch (e) {
                document.getElementById('config_msg').textContent = '错误: ' + e.message;
            }
        }

        async function deleteConfig() {
            var configName = document.getElementById('config_list').value;
            if (!configName) {
                document.getElementById('config_msg').textContent = '请选择配置';
                return;
            }

            try {
                var resp = await fetch('/delete_config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ config_name: configName })
                });
                var result = await resp.json();
                document.getElementById('config_msg').textContent = result.message;
                loadConfigList();
            } catch (e) {
                document.getElementById('config_msg').textContent = '错误: ' + e.message;
            }
        }

        function setSlider(id, value) {
            var slider = document.getElementById(id);
            if (slider) {
                slider.value = value;
                var span = document.getElementById(id + '_val');
                if (span) span.textContent = value;
            }
        }

        // Init
        loadConfigList();
    </script>
</body>
</html>"""


if __name__ == "__main__":
    model_path = os.environ.get("MODEL_PATH", "./models/Ming-omni-tts-0.5B")
    port = int(os.environ.get("PORT", 7860))
    load_model = os.environ.get("LOAD_MODEL", "true").lower() == "true"

    if load_model:
        load_model_fn(model_path)

    flask_app = create_webui(model_path, load_model=False, external_model=model)

    print(f"\n{'=' * 60}")
    print(f"Ming-Omni-TTS WebUI 已启动!")
    print(f"{'=' * 60}")
    print(f"WebUI:    http://localhost:{port}/")
    print(f"{'=' * 60}\n")

    flask_app.run(host="0.0.0.0", port=port, debug=False)
