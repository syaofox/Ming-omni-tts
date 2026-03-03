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
from flask import Flask, request, send_file, jsonify, render_template
from flask_cors import CORS
from loguru import logger

warnings.filterwarnings("ignore")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from common import (
    OUTPUT_DIR,
    CONFIG_DIR,
    UPLOAD_DIR,
    get_pinyin,
    get_pinyin_initials,
    to_traditional,
    save_config,
    get_config_list,
    load_config,
    delete_config,
)

from modeling_bailingmm import BailingMMNativeForConditionalGeneration
from sentence_manager.sentence_manager import SentenceNormalizer
from spkemb_extractor import SpkembExtractor


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

    def load_ip_data():
        ip_data_path = os.path.join(current_dir, "cookbooks", "ip_data.json")
        if os.path.exists(ip_data_path):
            with open(ip_data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    @app.route("/")
    def index():
        config_list = get_config_list()
        return render_template("webui.html", config_list=config_list)

    @app.route("/ip_list", methods=["GET"])
    def list_ip():
        ip_data = load_ip_data()
        return jsonify(list(ip_data.keys()))

    @app.route("/ip_data", methods=["GET"])
    def get_ip_data():
        ip_data = load_ip_data()
        ip_data_with_pinyin = {}
        for name, desc in ip_data.items():
            traditional = to_traditional(name)
            ip_data_with_pinyin[name] = {
                "description": desc,
                "pinyin": get_pinyin(name),
                "initials": get_pinyin_initials(name),
                "traditional": traditional,
                "traditional_pinyin": get_pinyin(traditional),
                "traditional_initials": get_pinyin_initials(traditional),
            }
        return jsonify(ip_data_with_pinyin)

    @app.route("/configs", methods=["GET"])
    def list_configs():
        configs = get_config_list()
        configs_with_pinyin = []
        for c in configs:
            configs_with_pinyin.append(
                {
                    "name": c,
                    "pinyin": get_pinyin(c),
                    "initials": get_pinyin_initials(c),
                }
            )
        return jsonify(configs_with_pinyin)

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
        ip = data.get("ip")

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
            ip=ip,
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
