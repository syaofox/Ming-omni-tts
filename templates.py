#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) Ant Group. All rights reserved.


def get_common_css():
    return """
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
    .reset-btn { background: #9E9E9E; color: white; border: none; border-radius: 4px; padding: 4px 8px; cursor: pointer; font-size: 14px; margin-left: 5px; }
    .reset-btn:hover { background: #757575; }
    .tooltip-label { position: relative; cursor: help; }
    .tooltip-label:hover::after {
        content: attr(data-tooltip);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background: #333;
        color: #fff;
        padding: 8px 12px;
        border-radius: 4px;
        font-size: 12px;
        white-space: nowrap;
        z-index: 1000;
        margin-bottom: 8px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    .tooltip-label:hover::before {
        content: '';
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        border: 6px solid transparent;
        border-top-color: #333;
        margin-bottom: -8px;
    }
    .search-dropdown {
        position: absolute;
        top: 100%;
        left: 0;
        right: 0;
        max-height: 200px;
        overflow-y: auto;
        background: white;
        border: 1px solid #ddd;
        border-radius: 4px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        z-index: 1000;
    }
    .search-dropdown div {
        padding: 10px 12px;
        cursor: pointer;
        border-bottom: 1px solid #f0f0f0;
    }
    .search-dropdown div:last-child {
        border-bottom: none;
    }
    .search-dropdown div:hover {
        background: #f5f5f5;
    }
    .search-dropdown div.no-result {
        color: #999;
        cursor: default;
    }
    """


def get_search_dropdown_js(
    data_var_name="dataList",
    search_input_id="search",
    dropdown_id="dropdown",
    hidden_input_id=None,
    onselect_callback=None,
):
    hidden_input_part = (
        f"""
    document.getElementById('{hidden_input_id}').value = name;
"""
        if hidden_input_id
        else ""
    )

    callback_part = (
        f"""
                {onselect_callback}(name);"""
        if onselect_callback
        else ""
    )

    return f"""
        function show{search_input_id.capitalize()}Dropdown(filter) {{
            var dropdown = document.getElementById('{dropdown_id}');
            var input = document.getElementById('{search_input_id}');
            dropdown.innerHTML = '';
            dropdown.style.display = 'block';
            
            var filterLower = filter.toLowerCase().trim();
            var filterNoSpace = filterLower.replace(/\\s+/g, '');
            
            var matched = {data_var_name}.filter(function(c) {{
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
                    dropdown.style.display = 'none';
                    {hidden_input_part}
                    {callback_part}
                }};
                dropdown.appendChild(div);
            }});
        }}
        
        document.getElementById('{search_input_id}').addEventListener('input', function() {{
            var value = this.value;
            if (!value) {{
                {f"document.getElementById('{hidden_input_id}').value = '';" if hidden_input_id else ""}
            }}
            if (value) {{
                show{search_input_id.capitalize()}Dropdown(value);
            }} else {{
                show{search_input_id.capitalize()}Dropdown('');
            }}
        }});
        
        document.getElementById('{search_input_id}').addEventListener('focus', function() {{
            show{search_input_id.capitalize()}Dropdown(this.value);
        }});
        
        document.addEventListener('click', function(e) {{
            var container = e.target.closest('.search-select-container');
            if (!container) {{
                document.getElementById('{dropdown_id}').style.display = 'none';
            }}
        }});
"""


def get_api_html(config_list_json, default_speaker="小缘", port=7860):
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
        <h1>Ming-Omni-TTS 语音合成</h1>
        <div class="info">
            <p><strong>API 调用方式：</strong></p>
            <code id="api-example">GET http://localhost:{port}/?text=要合成的文本&speaker={default_speaker}</code>
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
        <button onclick="generate()">生成语音</button>
        <div id="result"></div>
    </div>
    <script>
        var configDataList = {config_list_json};
        
        {get_search_dropdown_js("configDataList", "speaker_search", "speaker_dropdown", "speaker", "updateApiExample")}
        
        function updateApiExample() {{
            const speaker = document.getElementById('speaker').value;
            const example = 'http://localhost:{port}/?text=要合成的文本&speaker=' + encodeURIComponent(speaker);
            document.getElementById('api-example').textContent = example;
        }}
        
        if (configDataList.length > 0) {{
            document.getElementById('speaker').value = configDataList[0].name;
            document.getElementById('speaker_search').value = configDataList[0].name;
            updateApiExample();
        }}
        
        async function generate() {{
            const text = document.getElementById('text').value;
            const speaker = document.getElementById('speaker').value;
            const result = document.getElementById('result');
            
            if (!text) {{
                result.innerHTML = '<p style="color:red;">请输入文本</p>';
                return;
            }}
            
            result.innerHTML = '<p>正在生成...</p>';
            
            try {{
                const url = '/?text=' + encodeURIComponent(text) + '&speaker=' + encodeURIComponent(speaker);
                const response = await fetch(url);
                
                if (!response.ok) {{
                    const error = await response.text();
                    result.innerHTML = '<p style="color:red;">错误: ' + error + '</p>';
                    return;
                }}
                
                const blob = await response.blob();
                const audioUrl = URL.createObjectURL(blob);
                result.innerHTML = '<audio controls src="' + audioUrl + '"></audio>';
            }} catch (e) {{
                result.innerHTML = '<p style="color:red;">错误: ' + e.message + '</p>';
            }}
        }}
    </script>
</body>
</html>"""
