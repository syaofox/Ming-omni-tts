#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.

import json
import os
import time


def get_base_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    return base_dir


def generate_http_tts_config(
    ip: str = "10.10.10.10", port: int = 7861, output_path: str = None
) -> list:
    base_dir = get_base_dir()
    config_dir = os.path.join(base_dir, "saved_configs")

    subdirs = [
        d
        for d in os.listdir(config_dir)
        if os.path.isdir(os.path.join(config_dir, d)) and d != ".gitignore"
    ]

    subdirs_sorted = sorted(subdirs)

    configs = []
    base_timestamp = int(time.time() * 1000)

    for i, name in enumerate(subdirs_sorted):
        timestamp = base_timestamp + i
        configs.append(
            {
                "concurrentRate": "",
                "contentType": "",
                "enabledCookieJar": False,
                "header": "",
                "id": timestamp,
                "lastUpdateTime": timestamp,
                "loginCheckJs": "",
                "loginUi": "",
                "loginUrl": "",
                "name": f"Ming-omni-{name}",
                "url": f"http://{ip}:{port}/?text={{{{ speakText}}}}&speaker={name}",
            }
        )

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(configs, f, ensure_ascii=False, indent=2)

    return configs


if __name__ == "__main__":
    import sys

    ip = sys.argv[1] if len(sys.argv) > 1 else "10.10.10.10"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 7861

    base_dir = get_base_dir()
    output_path = os.path.join(base_dir, "saved_configs", "httpTts.json")

    configs = generate_http_tts_config(
        ip=ip,
        port=port,
        output_path=output_path,
    )

    print(f"已生成 {len(configs)} 个配置项")
    for c in configs:
        print(f"  - {c['name']}: {c['url']}")
