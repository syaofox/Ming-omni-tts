var ipData = {};
var configList = [];

async function loadIPData() {
    try {
        var resp = await fetch('/ip_data');
        ipData = await resp.json();
        initIPSelect();
        initZSIPSelect();
    } catch (e) {
        console.error('Failed to load IP data:', e);
    }
}

async function loadConfigList() {
    try {
        var resp = await fetch('/configs');
        configList = await resp.json();
        updateConfigDropdowns();
    } catch (e) {
        console.error('Failed to load config list:', e);
    }
}

function updateConfigDropdowns() {
    var instructSelect = document.getElementById('instruct_config_select');
    var zsSelect = document.getElementById('zs_config_select');
    
    var instructConfigs = configList.filter(function(c) {
        return c.task_type === 'Instruct TTS';
    });
    
    var zeroShotConfigs = configList.filter(function(c) {
        return c.task_type === 'Zero-shot TTS';
    });
    
    window.instructConfigs = instructConfigs;
    window.zeroShotConfigs = zeroShotConfigs;
}

function showConfigDropdown(inputId, dropdownId, hiddenId, configs) {
    var dropdown = document.getElementById(dropdownId);
    var input = document.getElementById(inputId);
    dropdown.innerHTML = '';
    dropdown.style.display = 'block';
    
    var filter = input.value || '';
    var filterLower = filter.toLowerCase().trim();
    var filterNoSpace = filterLower.replace(/\s+/g, '');
    
    var matched = configs.filter(function(c) {
        var nameLower = c.name.toLowerCase();
        var pinyinLower = (c.pinyin || '').toLowerCase();
        var initialsLower = (c.initials || '').toLowerCase();
        
        return nameLower.includes(filterLower) ||
               pinyinLower.includes(filterLower) ||
               initialsLower.includes(filterNoSpace);
    });
    
    if (matched.length === 0) {
        var noResult = document.createElement('div');
        noResult.className = 'no-result';
        noResult.textContent = '无匹配结果';
        dropdown.appendChild(noResult);
        return;
    }
    
    matched.forEach(function(c) {
        var div = document.createElement('div');
        div.textContent = c.name + ' (' + (c.pinyin || '') + ')';
        div.onclick = function(e) {
            e.stopPropagation();
            input.value = c.name;
            dropdown.style.display = 'none';
            document.getElementById(hiddenId).value = c.name;
        };
        dropdown.appendChild(div);
    });
}

function hideConfigDropdown() {
    document.getElementById('instruct_config_dropdown').style.display = 'none';
    document.getElementById('zs_config_dropdown').style.display = 'none';
}

document.getElementById('instruct_config_search').addEventListener('input', function() {
    showConfigDropdown('instruct_config_search', 'instruct_config_dropdown', 'instruct_config_select', window.instructConfigs || []);
});

document.getElementById('instruct_config_search').addEventListener('focus', function() {
    showConfigDropdown('instruct_config_search', 'instruct_config_dropdown', 'instruct_config_select', window.instructConfigs || []);
});

document.getElementById('zs_config_search').addEventListener('input', function() {
    showConfigDropdown('zs_config_search', 'zs_config_dropdown', 'zs_config_select', window.zeroShotConfigs || []);
});

document.getElementById('zs_config_search').addEventListener('focus', function() {
    showConfigDropdown('zs_config_search', 'zs_config_dropdown', 'zs_config_select', window.zeroShotConfigs || []);
});

document.addEventListener('click', function(e) {
    var container = e.target.closest('.search-select-container');
    if (!container) {
        hideConfigDropdown();
    }
});

async function loadInstructConfig() {
    var configName = document.getElementById('instruct_config_select').value;
    if (!configName) {
        alert('请选择要加载的配置');
        return;
    }
    
    try {
        var resp = await fetch('/load_config?config_name=' + encodeURIComponent(configName));
        var result = await resp.json();
        if (result.success) {
            var data = result.data;
            if (data.task_type !== 'Instruct TTS') {
                alert('该配置不是 Instruct TTS 类型的配置');
                return;
            }
            if (data.instruct_type) {
                document.getElementById('instruct_type').value = data.instruct_type;
                updateInstructVisibility();
            }
            if (data.prompt_audio) {
                displayConfigAudio(data.prompt_audio, 'instruct_prompt_audio_display');
                document.getElementById('instruct_config_audio_path').value = data.prompt_audio;
            } else {
                clearConfigAudio('instruct_prompt_audio_display');
                document.getElementById('instruct_config_audio_path').value = '';
            }
            if (data.emotion) {
                document.getElementById('instruct_emotion').value = data.emotion;
            }
            if (data.dialect) {
                document.getElementById('instruct_dialect').value = data.dialect;
            }
            if (data.ip) {
                document.getElementById('ip_search').value = data.ip;
                document.getElementById('instruct_ip').value = data.ip;
            }
            if (data.style) {
                document.getElementById('instruct_style').value = data.style;
            }
            if (data.speech_speed) {
                var speedMap = {0.7: '慢速', 1.0: '中速', 1.3: '快速'};
                document.getElementById('instruct_speed').value = speedMap[data.speech_speed] || '中速';
            }
            if (data.pitch) {
                var pitchMap = {0.8: '低', 1.0: '中', 1.2: '高'};
                document.getElementById('instruct_pitch').value = pitchMap[data.pitch] || '中';
            }
            if (data.volume) {
                var volumeMap = {0.7: '低', 1.0: '中', 1.3: '高'};
                document.getElementById('instruct_volume').value = volumeMap[data.volume] || '中';
            }
            showConfigMessage('instruct_config_msg', '配置 "' + configName + '" 已加载', false);
        } else {
            showConfigMessage('instruct_config_msg', '加载失败: ' + result.message, true);
        }
    } catch (e) {
        showConfigMessage('instruct_config_msg', '加载失败: ' + e.message, true);
    }
}

async function loadZeroShotConfig() {
    var configName = document.getElementById('zs_config_select').value;
    if (!configName) {
        showConfigMessage('zs_config_msg', '请选择要加载的配置', true);
        return;
    }
    
    try {
        var resp = await fetch('/load_config?config_name=' + encodeURIComponent(configName));
        var result = await resp.json();
        if (result.success) {
            var data = result.data;
            if (data.task_type !== 'Zero-shot TTS') {
                showConfigMessage('zs_config_msg', '该配置不是 Zero-shot TTS 类型的配置', true);
                return;
            }
            if (data.prompt_audio) {
                displayConfigAudio(data.prompt_audio, 'zs_prompt_audio_display');
                document.getElementById('zs_config_audio_path').value = data.prompt_audio;
            } else {
                clearConfigAudio('zs_prompt_audio_display');
                document.getElementById('zs_config_audio_path').value = '';
            }
            if (data.ip) {
                document.getElementById('zs_ip_search').value = data.ip;
                document.getElementById('zs_instruct_ip').value = data.ip;
            }
            if (data.prompt_text) {
                document.getElementById('zs_prompt_text').value = data.prompt_text;
            }
            showConfigMessage('zs_config_msg', '配置 "' + configName + '" 已加载', false);
        } else {
            showConfigMessage('zs_config_msg', '加载失败: ' + result.message, true);
        }
    } catch (e) {
        showConfigMessage('zs_config_msg', '加载失败: ' + e.message, true);
    }
}

function showConfigMessage(msgId, message, isError) {
    var container = document.getElementById('toast-container');
    var toast = document.createElement('div');
    toast.className = 'toast ' + (isError ? 'error' : 'success');
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(function() {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        toast.style.transition = 'all 0.3s ease-out';
        setTimeout(function() {
            container.removeChild(toast);
        }, 300);
    }, 3000);
}

function showSaveInstructConfigModal() {
    console.log('showSaveInstructConfigModal called');
    var configName = document.getElementById('instruct_config_search').value;
    console.log('configName from search input:', configName);
    if (!configName || !configName.trim()) {
        showConfigMessage('instruct_config_msg', '请在搜索框输入配置名称', true);
        return;
    }
    saveInstructConfig(configName.trim());
}

async function saveInstructConfig(configName) {
    var instructType = document.getElementById('instruct_type').value;
    var promptAudio = null;
    
    console.log('saveInstructConfig called, instructType:', instructType);
    
    if (instructType !== 'ip' && instructType !== 'style') {
        var fileInput = document.getElementById('instruct_prompt_audio');
        if (fileInput && fileInput.files && fileInput.files[0]) {
            promptAudio = await uploadAudioIfNeeded('instruct_prompt_audio');
            console.log('Uploaded new audio:', promptAudio);
        }
        if (!promptAudio) {
            var savedAudio = document.getElementById('instruct_config_audio_path').value;
            console.log('Saved audio path from hidden input:', savedAudio);
            promptAudio = savedAudio || null;
        }
    }
    
    console.log('Final promptAudio to save:', promptAudio);

    var emotion = '无';
    var dialect = '无';
    var style = '无';
    var ip = null;
    var speechSpeed = 1.0;
    var pitch = 1.0;
    var volume = 1.0;

    if (instructType === 'emotion') {
        emotion = document.getElementById('instruct_emotion').value;
    } else if (instructType === 'dialect') {
        dialect = document.getElementById('instruct_dialect').value;
    } else if (instructType === 'ip') {
        ip = document.getElementById('instruct_ip').value;
    } else if (instructType === 'style') {
        style = document.getElementById('instruct_style').value;
    } else if (instructType === 'basic') {
        var speedMap = {'慢速': 0.7, '中速': 1.0, '快速': 1.3};
        var pitchMap = {'低': 0.8, '中': 1.0, '高': 1.2};
        var volumeMap = {'低': 0.7, '中': 1.0, '高': 1.3};
        speechSpeed = speedMap[document.getElementById('instruct_speed').value];
        pitch = pitchMap[document.getElementById('instruct_pitch').value];
        volume = volumeMap[document.getElementById('instruct_volume').value];
    }

    var data = {
        config_name: configName,
        task_type: 'Instruct TTS',
        instruct_type: instructType,
        prompt_audio: promptAudio,
        emotion: emotion,
        dialect: dialect,
        style: style,
        ip: ip,
        speech_speed: speechSpeed,
        pitch: pitch,
        volume: volume,
    };

    console.log('Saving config data:', JSON.stringify(data, null, 2));

    try {
        var resp = await fetch('/save_config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        var result = await resp.json();
        if (result.success) {
            showConfigMessage('instruct_config_msg', result.message, false);
            loadConfigList();
        } else {
            showConfigMessage('instruct_config_msg', '保存失败: ' + result.message, true);
        }
    } catch (e) {
        showConfigMessage('instruct_config_msg', '保存失败: ' + e.message, true);
    }
}

function showSaveZeroShotConfigModal() {
    console.log('showSaveZeroShotConfigModal called');
    var configName = document.getElementById('zs_config_search').value;
    console.log('zs configName:', configName);
    if (!configName || !configName.trim()) {
        showConfigMessage('zs_config_msg', '请在搜索框输入配置名称', true);
        return;
    }
    saveZeroShotConfig(configName.trim());
}

async function saveZeroShotConfig(configName) {
    console.log('saveZeroShotConfig called');
    var fileInput = document.getElementById('zs_prompt_audio');
    var promptAudio = null;
    if (fileInput && fileInput.files && fileInput.files[0]) {
        promptAudio = await uploadAudioIfNeeded('zs_prompt_audio');
    }
    if (!promptAudio) {
        var savedAudio = document.getElementById('zs_config_audio_path').value;
        console.log('zs_config_audio_path value:', savedAudio);
        promptAudio = savedAudio || null;
    }

    var ip = document.getElementById('zs_instruct_ip').value;
    var promptText = document.getElementById('zs_prompt_text').value || null;

    var data = {
        config_name: configName,
        task_type: 'Zero-shot TTS',
        prompt_audio: promptAudio,
        prompt_text: promptText,
        ip: ip || null,
    };

    console.log('Saving Zero-shot config:', JSON.stringify(data, null, 2));

    try {
        var resp = await fetch('/save_config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        var result = await resp.json();
        if (result.success) {
            showConfigMessage('zs_config_msg', result.message, false);
            loadConfigList();
        } else {
            showConfigMessage('zs_config_msg', '保存失败: ' + result.message, true);
        }
    } catch (e) {
        showConfigMessage('zs_config_msg', '保存失败: ' + e.message, true);
    }
}

var ipList = [];

function initIPSelect() {
    ipList = Object.keys(ipData).sort();
}

function initZSIPSelect() {
    // Zero-shot IP uses same data
}

function showIPDropdown(inputId, dropdownId, hiddenId) {
    var dropdown = document.getElementById(dropdownId);
    var input = document.getElementById(inputId);
    dropdown.innerHTML = '';
    dropdown.style.display = 'block';
    
    var filter = input.value || '';
    var filterLower = filter.toLowerCase().trim();
    var filterNoSpace = filterLower.replace(/\s+/g, '');
    
    var matched = ipList.filter(function(ip) {
        var data = ipData[ip] || {};
        var nameLower = ip.toLowerCase();
        var pinyinLower = (data.pinyin || '').toLowerCase();
        var initialsLower = (data.initials || '').toLowerCase();
        
        return nameLower.includes(filterLower) ||
               pinyinLower.includes(filterLower) ||
               initialsLower.includes(filterNoSpace);
    });
    
    if (matched.length === 0) {
        var noResult = document.createElement('div');
        noResult.className = 'no-result';
        noResult.textContent = '无匹配结果';
        dropdown.appendChild(noResult);
        return;
    }
    
    matched.forEach(function(ip) {
        var div = document.createElement('div');
        div.textContent = ip;
        div.onclick = function(e) {
            e.stopPropagation();
            input.value = ip;
            dropdown.style.display = 'none';
            document.getElementById(hiddenId).value = ip;
        };
        dropdown.appendChild(div);
    });
}

function hideIPDropdown() {
    document.getElementById('ip_dropdown').style.display = 'none';
    document.getElementById('zs_ip_dropdown').style.display = 'none';
}

document.getElementById('ip_search').addEventListener('input', function() {
    showIPDropdown('ip_search', 'ip_dropdown', 'instruct_ip');
});

document.getElementById('ip_search').addEventListener('focus', function() {
    showIPDropdown('ip_search', 'ip_dropdown', 'instruct_ip');
});

document.getElementById('zs_ip_search').addEventListener('input', function() {
    showIPDropdown('zs_ip_search', 'zs_ip_dropdown', 'zs_instruct_ip');
});

document.getElementById('zs_ip_search').addEventListener('focus', function() {
    showIPDropdown('zs_ip_search', 'zs_ip_dropdown', 'zs_instruct_ip');
});

document.addEventListener('click', function(e) {
    var container = e.target.closest('.search-select-container');
    if (!container) {
        hideIPDropdown();
    }
});

function updateInstructVisibility() {
    var instructType = document.getElementById('instruct_type').value;
    
    var promptGroup = document.getElementById('instruct_prompt_group');
    var emotionControls = document.getElementById('emotion_controls');
    var dialectControls = document.getElementById('dialect_controls');
    var ipControls = document.getElementById('ip_controls');
    var styleControls = document.getElementById('style_controls');
    var basicControls = document.getElementById('basic_controls');
    
    if (instructType === 'ip' || instructType === 'style') {
        promptGroup.style.display = 'none';
    } else {
        promptGroup.style.display = 'block';
    }
    
    emotionControls.style.display = instructType === 'emotion' ? 'block' : 'none';
    dialectControls.style.display = instructType === 'dialect' ? 'block' : 'none';
    ipControls.style.display = instructType === 'ip' ? 'block' : 'none';
    styleControls.style.display = instructType === 'style' ? 'block' : 'none';
    basicControls.style.display = instructType === 'basic' ? 'block' : 'none';
}

function initDropZone(dropZoneId, fileInputId, displayId) {
    var dropZone = document.getElementById(dropZoneId);
    var fileInput = document.getElementById(fileInputId);
    
    if (!dropZone || !fileInput) return;
    
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
    
    fileInput.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            var file = this.files[0];
            var audioDisplay = document.getElementById(displayId);
            var url = URL.createObjectURL(file);
            audioDisplay.innerHTML = '<audio controls src="' + url + '"></audio><p style="font-size:12px;color:#666;">' + file.name + '</p>';
        }
    });
}

function displayConfigAudio(audioPath, displayId) {
    console.log('displayConfigAudio called:', audioPath, displayId);
    var audioDisplay = document.getElementById(displayId);
    if (!audioDisplay) {
        console.error('Audio display element not found:', displayId);
        return;
    }
    if (audioPath) {
        var match = audioPath.match(/saved_configs[/\\]([^/\\]+)/);
        console.log('Regex match:', match);
        var configName = match ? match[1] : null;
        console.log('Extracted configName:', configName);
        if (configName) {
            audioDisplay.innerHTML = '<audio controls src="/config_audio/' + encodeURIComponent(configName) + '"></audio><p style="font-size:12px;color:#666;">已加载参考音频</p>';
        } else {
            audioDisplay.innerHTML = '<p style="font-size:12px;color:#666;">音频路径无效: ' + audioPath + '</p>';
        }
    } else {
        audioDisplay.innerHTML = '';
    }
}

function clearConfigAudio(displayId) {
    var audioDisplay = document.getElementById(displayId);
    audioDisplay.innerHTML = '';
}

initDropZone('instruct_drop_zone', 'instruct_prompt_audio', 'instruct_prompt_audio_display');
initDropZone('zs_drop_zone', 'zs_prompt_audio', 'zs_prompt_audio_display');
initDropZone('pod_drop_zone1', 'pod_prompt_audio1', 'pod_prompt_audio_display1');
initDropZone('pod_drop_zone2', 'pod_prompt_audio2', 'pod_prompt_audio_display2');
initDropZone('pod_drop_zone3', 'pod_prompt_audio3', 'pod_prompt_audio_display3');
initDropZone('swb_drop_zone', 'swb_prompt_audio', 'swb_prompt_audio_display');

document.querySelectorAll('input[type="range"]').forEach(function(slider) {
    slider.addEventListener('input', function() {
        var span = document.getElementById(this.id + '_val');
        if (span) span.textContent = this.value;
    });
});

function switchTab(tabId) {
    document.querySelectorAll('.tab').forEach(function(t) { t.classList.remove('active'); });
    document.querySelectorAll('.tab-content').forEach(function(c) { c.classList.remove('active'); });
    document.querySelector('.tab[onclick="switchTab(\'' + tabId + '\')"]').classList.add('active');
    document.getElementById(tabId).classList.add('active');
}

function showResult(id, success, message, audioUrl) {
    showResultCommon(id + '_result', success, message, audioUrl);
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

async function generateInstructTTS() {
    var btn = document.getElementById('instruct_generate');
    btn.disabled = true;
    
    var startTime = Date.now();
    var timer = null;
    var resultDiv = document.getElementById('instruct_result');
    resultDiv.className = 'result';
    resultDiv.innerHTML = '<p>生成中... <span id="instruct_elapsed">0.0</span> 秒</span></p>';
    
    timer = setInterval(function() {
        var elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        var elapsedSpan = document.getElementById('instruct_elapsed');
        if (elapsedSpan) elapsedSpan.textContent = elapsed;
    }, 100);

    var text = document.getElementById('instruct_text').value;
    if (!text) {
        clearInterval(timer);
        showResult('instruct', false, '请输入文本');
        btn.disabled = false;
        btn.textContent = '生成语音';
        return;
    }

    var instructType = document.getElementById('instruct_type').value;
    var promptAudio = null;
    
    if (instructType !== 'ip' && instructType !== 'style') {
        promptAudio = await uploadAudioIfNeeded('instruct_prompt_audio');
        if (!promptAudio) {
            promptAudio = document.getElementById('instruct_config_audio_path').value || null;
        }
    }

    var emotion = '无';
    var dialect = '无';
    var style = '无';
    var ip = null;
    var speechSpeed = 1.0;
    var pitch = 1.0;
    var volume = 1.0;

    if (instructType === 'emotion') {
        emotion = document.getElementById('instruct_emotion').value;
    } else if (instructType === 'dialect') {
        dialect = document.getElementById('instruct_dialect').value;
    } else if (instructType === 'ip') {
        ip = document.getElementById('instruct_ip').value;
    } else if (instructType === 'style') {
        style = document.getElementById('instruct_style').value;
    } else if (instructType === 'basic') {
        var speedMap = {'慢速': 0.7, '中速': 1.0, '快速': 1.3};
        var pitchMap = {'低': 0.8, '中': 1.0, '高': 1.2};
        var volumeMap = {'低': 0.7, '中': 1.0, '高': 1.3};
        speechSpeed = speedMap[document.getElementById('instruct_speed').value];
        pitch = pitchMap[document.getElementById('instruct_pitch').value];
        volume = volumeMap[document.getElementById('instruct_volume').value];
    }

    var data = {
        task_type: 'Instruct TTS',
        text: text,
        prompt_audio: promptAudio,
        emotion: emotion,
        dialect: dialect,
        style: style,
        ip: ip,
        speech_speed: speechSpeed,
        pitch: pitch,
        volume: volume,
        seed: parseInt(document.getElementById('settings_seed').value) || null,
    };

    try {
        var resp = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        var result = await resp.json();
        clearInterval(timer);
        var totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
        if (result.success) {
            showResult('instruct', true, result.message + ' (用时: ' + totalTime + '秒)', result.audio_url);
        } else {
            showResult('instruct', false, result.message);
        }
    } catch (e) {
        clearInterval(timer);
        showResult('instruct', false, '错误: ' + e.message);
    }

    btn.disabled = false;
    btn.textContent = '生成语音';
}

async function generateZeroShotTTS() {
    var btn = document.getElementById('zs_generate');
    btn.disabled = true;
    
    var startTime = Date.now();
    var timer = null;
    var resultDiv = document.getElementById('zs_result');
    resultDiv.className = 'result';
    resultDiv.innerHTML = '<p>生成中... <span id="zs_elapsed">0.0</span> 秒</span></p>';
    
    timer = setInterval(function() {
        var elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        var elapsedSpan = document.getElementById('zs_elapsed');
        if (elapsedSpan) elapsedSpan.textContent = elapsed;
    }, 100);

    var text = document.getElementById('zs_text').value;
    var promptAudio = await uploadAudioIfNeeded('zs_prompt_audio');
    if (!promptAudio) {
        promptAudio = document.getElementById('zs_config_audio_path').value || null;
    }
    var ip = document.getElementById('zs_instruct_ip').value;
    var promptText = document.getElementById('zs_prompt_text').value || null;

    if (!text) {
        clearInterval(timer);
        showResult('zs', false, '请输入文本');
        btn.disabled = false;
        return;
    }

    if (!promptAudio && !ip) {
        clearInterval(timer);
        showResult('zs', false, '请上传参考音频或选择内置人物');
        btn.disabled = false;
        return;
    }

    var data = {
        task_type: '零样本语音合成 (Zero-shot TTS)',
        text: text,
        prompt_audio: promptAudio,
        prompt_text: promptText,
        ip: ip || null,
        seed: parseInt(document.getElementById('settings_seed').value) || null,
    };

    try {
        var resp = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        var result = await resp.json();
        clearInterval(timer);
        var totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
        if (result.success) {
            showResult('zs', true, result.message + ' (用时: ' + totalTime + '秒)', result.audio_url);
        } else {
            showResult('zs', false, result.message);
        }
    } catch (e) {
        clearInterval(timer);
        showResult('zs', false, '错误: ' + e.message);
    }

    btn.disabled = false;
    btn.textContent = '克隆音色并生成语音';
}

async function generatePodcast() {
    var btn = document.getElementById('pod_generate');
    btn.disabled = true;
    
    var startTime = Date.now();
    var timer = null;
    var resultDiv = document.getElementById('pod_result');
    resultDiv.className = 'result';
    resultDiv.innerHTML = '<p>生成中... <span id="pod_elapsed">0.0</span> 秒</p>';
    
    timer = setInterval(function() {
        var elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        var elapsedSpan = document.getElementById('pod_elapsed');
        if (elapsedSpan) elapsedSpan.textContent = elapsed;
    }, 100);

    var text = document.getElementById('pod_text').value;
    if (!text) {
        clearInterval(timer);
        showResult('pod', false, '请输入对话脚本');
        btn.disabled = false;
        return;
    }

    var promptAudios = [];
    for (var i = 1; i <= 3; i++) {
        var audio = await uploadAudioIfNeeded('pod_prompt_audio' + i);
        if (audio) {
            promptAudios.push(audio);
        }
    }

    if (promptAudios.length < 2) {
        clearInterval(timer);
        showResult('pod', false, '请至少上传两个说话人的参考音频');
        btn.disabled = false;
        return;
    }

    var data = {
        task_type: 'Podcast',
        text: text,
        prompt_audio: promptAudios,
        seed: parseInt(document.getElementById('settings_seed').value) || null,
    };

    try {
        var resp = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        var result = await resp.json();
        clearInterval(timer);
        var totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
        if (result.success) {
            showResult('pod', true, result.message + ' (用时: ' + totalTime + '秒)', result.audio_url);
        } else {
            showResult('pod', false, result.message);
        }
    } catch (e) {
        clearInterval(timer);
        showResult('pod', false, '错误: ' + e.message);
    }

    btn.disabled = false;
}

async function generateSpeechWithBGM() {
    var btn = document.getElementById('swb_generate');
    btn.disabled = true;
    
    var startTime = Date.now();
    var timer = null;
    var resultDiv = document.getElementById('swb_result');
    resultDiv.className = 'result';
    resultDiv.innerHTML = '<p>生成中... <span id="swb_elapsed">0.0</span> 秒</p>';
    
    timer = setInterval(function() {
        var elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        var elapsedSpan = document.getElementById('swb_elapsed');
        if (elapsedSpan) elapsedSpan.textContent = elapsed;
    }, 100);

    var text = document.getElementById('swb_text').value;
    if (!text) {
        clearInterval(timer);
        showResult('swb', false, '请输入语音文本');
        btn.disabled = false;
        return;
    }

    var promptAudio = await uploadAudioIfNeeded('swb_prompt_audio');
    if (!promptAudio) {
        clearInterval(timer);
        showResult('swb', false, '请上传说话人参考音频');
        btn.disabled = false;
        return;
    }

    var genre = document.getElementById('swb_genre').value;
    var mood = document.getElementById('swb_mood').value;
    var instrument = document.getElementById('swb_instrument').value;
    var theme = document.getElementById('swb_theme').value;
    var snr = parseFloat(document.getElementById('swb_snr').value);

    var data = {
        task_type: 'Speech with BGM',
        text: text,
        prompt_audio: promptAudio,
        genre: genre,
        mood: mood,
        instrument: instrument,
        theme: theme,
        snr: snr,
        seed: parseInt(document.getElementById('settings_seed').value) || null,
    };

    try {
        var resp = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        var result = await resp.json();
        clearInterval(timer);
        var totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
        if (result.success) {
            showResult('swb', true, result.message + ' (用时: ' + totalTime + '秒)', result.audio_url);
        } else {
            showResult('swb', false, result.message);
        }
    } catch (e) {
        clearInterval(timer);
        showResult('swb', false, '错误: ' + e.message);
    }

    btn.disabled = false;
}

async function generateBGM() {
    var startTime = Date.now();
    var timer = null;
    var resultDiv = document.getElementById('bgm_result');
    resultDiv.className = 'result';
    resultDiv.innerHTML = '<p>生成中... <span id="bgm_elapsed">0.0</span> 秒</p>';
    
    timer = setInterval(function() {
        var elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        var elapsedSpan = document.getElementById('bgm_elapsed');
        if (elapsedSpan) elapsedSpan.textContent = elapsed;
    }, 100);

    var genre = document.getElementById('bgm_genre').value;
    var mood = document.getElementById('bgm_mood').value;
    var instrument = document.getElementById('bgm_instrument').value;
    var theme = document.getElementById('bgm_theme').value;
    var duration = document.getElementById('bgm_duration').value;

    var text = 'Genre: ' + genre + '. Mood: ' + mood + '. Instrument: ' + instrument + '. Theme: ' + theme + '. Duration: ' + duration + 's.';

    var data = {
        task_type: 'BGM Generation',
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
        seed: parseInt(document.getElementById('settings_seed').value) || null,
    };

    try {
        var resp = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        var result = await resp.json();
        clearInterval(timer);
        var totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
        if (result.success) {
            showResult('bgm', true, result.message + ' (用时: ' + totalTime + '秒)', result.audio_url);
        } else {
            showResult('bgm', false, result.message);
        }
    } catch (e) {
        clearInterval(timer);
        showResult('bgm', false, '错误: ' + e.message);
    }
}

async function generateTTA() {
    var startTime = Date.now();
    var timer = null;
    var resultDiv = document.getElementById('tta_result');
    resultDiv.className = 'result';
    resultDiv.innerHTML = '<p>生成中... <span id="tta_elapsed">0.0</span> 秒</p>';
    
    timer = setInterval(function() {
        var elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        var elapsedSpan = document.getElementById('tta_elapsed');
        if (elapsedSpan) elapsedSpan.textContent = elapsed;
    }, 100);

    var text = document.getElementById('tta_text').value;
    if (!text) {
        clearInterval(timer);
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
        seed: parseInt(document.getElementById('settings_seed').value) || null,
    };

    try {
        var resp = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });
        var result = await resp.json();
        clearInterval(timer);
        var totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
        if (result.success) {
            showResult('tta', true, result.message + ' (用时: ' + totalTime + '秒)', result.audio_url);
        } else {
            showResult('tta', false, result.message);
        }
    } catch (e) {
        clearInterval(timer);
        showResult('tta', false, '错误: ' + e.message);
    }
}

loadIPData();
loadConfigList();

function openSettings() {
    document.getElementById('settingsModal').style.display = 'block';
    loadSeed();
}

function closeSettings() {
    document.getElementById('settingsModal').style.display = 'none';
}

async function loadSeed() {
    try {
        var resp = await fetch('/seed');
        var result = await resp.json();
        if (result.success) {
            document.getElementById('settings_seed').value = result.seed;
        }
    } catch (e) {
        console.error('Failed to load seed:', e);
    }
}

async function saveSeed() {
    var seedInput = document.getElementById('settings_seed').value;
    var seedValue = seedInput ? parseInt(seedInput) : null;
    
    try {
        var resp = await fetch('/seed', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ seed: seedValue })
        });
        var result = await resp.json();
        if (result.success) {
            document.getElementById('settings_msg').textContent = 'Seed已保存: ' + result.seed;
            document.getElementById('settings_seed').value = result.seed;
        } else {
            document.getElementById('settings_msg').textContent = 'Error: ' + result.message;
        }
    } catch (e) {
        document.getElementById('settings_msg').textContent = 'Error: ' + e.message;
    }
}

async function randomizeSeed() {
    document.getElementById('settings_seed').value = '';
    await saveSeed();
}

window.onclick = function(event) {
    var modal = document.getElementById('settingsModal');
    if (event.target == modal) {
        closeSettings();
    }
}
