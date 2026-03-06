var ipData = {};
var configList = [];
var ipList = [];

async function loadIPData() {
    try {
        var resp = await fetch('/ip_data');
        ipData = await resp.json();
        initIPSelect();
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
    var instructConfigs = configList.filter(function(c) {
        return c.task_type === 'Instruct TTS';
    });
    
    var zeroShotConfigs = configList.filter(function(c) {
        return c.task_type === 'Zero-shot TTS';
    });
    
    var instructSelect = document.getElementById('instruct_config_select');
    var zsSelect = document.getElementById('zs_config_select');
    
    if (instructSelect && instructSelect.setData) {
        instructSelect.setData(instructConfigs.map(function(c) {
            return { name: c.name, value: c.name };
        }));
    }
    
    if (zsSelect && zsSelect.setData) {
        zsSelect.setData(zeroShotConfigs.map(function(c) {
            return { name: c.name, value: c.name };
        }));
    }
}

function initIPSelect() {
    ipList = Object.keys(ipData).sort().map(function(ip) {
        return { name: ip, value: ip, pinyin: (ipData[ip] || {}).pinyin, initials: (ipData[ip] || {}).initials };
    });
    
    var instructIpSelect = document.getElementById('instruct_ip_select');
    var zsIpSelect = document.getElementById('zs_ip_select');
    
    if (instructIpSelect && instructIpSelect.setData) {
        instructIpSelect.setData(ipList);
    }
    
    if (zsIpSelect && zsIpSelect.setData) {
        zsIpSelect.setData(ipList);
    }
}

async function loadInstructConfig() {
    var select = document.getElementById('instruct_config_select');
    var configName = select ? select.getValue() : null;
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
            } else {
                document.getElementById('instruct_type').value = 'emotion';
                updateInstructVisibility();
            }
            if (data.prompt_audio) {
                displayConfigAudio(data.prompt_audio, 'instruct_audio_uploader');
            } else {
                clearConfigAudio('instruct_audio_uploader');
            }
            if (data.emotion) {
                document.getElementById('instruct_emotion').value = data.emotion;
            } else {
                document.getElementById('instruct_emotion').value = '高兴';
            }
            if (data.dialect) {
                document.getElementById('instruct_dialect').value = data.dialect;
            } else {
                document.getElementById('instruct_dialect').value = '普通话';
            }
            if (data.ip) {
                var ipSelect = document.getElementById('instruct_ip_select');
                if (ipSelect && ipSelect.setValue) ipSelect.setValue(data.ip);
            } else {
                var ipSelect = document.getElementById('instruct_ip_select');
                if (ipSelect && ipSelect.clear) ipSelect.clear();
            }
            if (data.style) {
                document.getElementById('instruct_style').value = data.style;
            } else {
                document.getElementById('instruct_style').value = '';
            }
            if (data.speech_speed) {
                var speedMap = {0.7: '慢速', 1.0: '中速', 1.3: '快速'};
                document.getElementById('instruct_speed').value = speedMap[data.speech_speed] || '中速';
            } else {
                document.getElementById('instruct_speed').value = '中速';
            }
            if (data.pitch) {
                var pitchMap = {0.8: '低', 1.0: '中', 1.2: '高'};
                document.getElementById('instruct_pitch').value = pitchMap[data.pitch] || '中';
            } else {
                document.getElementById('instruct_pitch').value = '中';
            }
            if (data.volume) {
                var volumeMap = {0.7: '低', 1.0: '中', 1.3: '高'};
                document.getElementById('instruct_volume').value = volumeMap[data.volume] || '中';
            } else {
                document.getElementById('instruct_volume').value = '中';
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
    var select = document.getElementById('zs_config_select');
    var configName = select ? select.getValue() : null;
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
                displayConfigAudio(data.prompt_audio, 'zs_audio_uploader');
            } else {
                clearConfigAudio('zs_audio_uploader');
            }
            if (data.ip) {
                var ipSelect = document.getElementById('zs_ip_select');
                if (ipSelect && ipSelect.setValue) ipSelect.setValue(data.ip);
            } else {
                var ipSelect = document.getElementById('zs_ip_select');
                if (ipSelect && ipSelect.clear) ipSelect.clear();
            }
            if (data.prompt_text) {
                document.getElementById('zs_prompt_text').value = data.prompt_text;
            } else {
                document.getElementById('zs_prompt_text').value = '';
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
    showToast(message, isError);
}

function showSaveInstructConfigModal() {
    var select = document.getElementById('instruct_config_select');
    var configName = select ? select.getValue() : null;
    if (!configName || !configName.trim()) {
        showConfigMessage('instruct_config_msg', '请先选择或输入配置名称', true);
        return;
    }
    saveInstructConfig(configName.trim());
}

async function saveInstructConfig(configName) {
    var instructType = document.getElementById('instruct_type').value;
    var promptAudio = null;
    
    if (instructType !== 'ip' && instructType !== 'style') {
        var uploader = document.getElementById('instruct_audio_uploader');
        promptAudio = await uploadAudioIfNeeded('instruct_audio_uploader');
        if (!promptAudio) {
            promptAudio = uploader ? uploader.getValue() : null;
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
        ip = document.getElementById('instruct_ip_select').getValue();
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
    var select = document.getElementById('zs_config_select');
    var configName = select ? select.getValue() : null;
    if (!configName || !configName.trim()) {
        showConfigMessage('zs_config_msg', '请先选择或输入配置名称', true);
        return;
    }
    saveZeroShotConfig(configName.trim());
}

async function saveZeroShotConfig(configName) {
    var promptAudio = await uploadAudioIfNeeded('zs_audio_uploader');
    if (!promptAudio) {
        var uploader = document.getElementById('zs_audio_uploader');
        promptAudio = uploader ? uploader.getValue() : null;
    }

    var ip = document.getElementById('zs_ip_select').getValue();
    var promptText = document.getElementById('zs_prompt_text').value || null;

    var data = {
        config_name: configName,
        task_type: 'Zero-shot TTS',
        prompt_audio: promptAudio,
        prompt_text: promptText,
        ip: ip || null,
    };

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

function displayConfigAudio(audioPath, uploaderId) {
    var uploader = uploaderId ? document.getElementById(uploaderId) : null;
    if (!uploader || !uploader.setAudioPath) {
        console.error('Audio uploader element not found or invalid:', uploaderId);
        return;
    }
    if (audioPath) {
        var match = audioPath.match(/saved_configs[/\\]([^/\\]+)/);
        var configName = match ? match[1] : null;
        uploader.setAudioPath(audioPath, configName);
    } else {
        uploader.clear();
    }
}

function clearConfigAudio(uploaderId) {
    var uploader = uploaderId ? document.getElementById(uploaderId) : null;
    if (uploader && uploader.clear) {
        uploader.clear();
    }
}

function getAudioUploader(uploaderId) {
    return document.getElementById(uploaderId);
}

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

function disablePodcastInputs(disabled) {
    document.getElementById('pod_text').disabled = disabled;
    for (var i = 1; i <= 3; i++) {
        var uploader = document.getElementById('pod_audio_' + i);
        if (uploader) uploader.setDisabled(disabled);
    }
}

function disableInstructInputs(disabled) {
    document.getElementById('instruct_text').disabled = disabled;
    var uploader = document.getElementById('instruct_audio_uploader');
    if (uploader) uploader.setDisabled(disabled);
}

function disableZeroShotInputs(disabled) {
    document.getElementById('zs_text').disabled = disabled;
    document.getElementById('zs_prompt_text').disabled = disabled;
    var uploader = document.getElementById('zs_audio_uploader');
    if (uploader) uploader.setDisabled(disabled);
}

function disableSpeechWithBGMInputs(disabled) {
    document.getElementById('swb_text').disabled = disabled;
    var uploader = document.getElementById('swb_audio');
    if (uploader) uploader.setDisabled(disabled);
    var ids = ['swb_genre', 'swb_mood', 'swb_instrument', 'swb_theme', 'swb_snr'];
    ids.forEach(function(id) {
        var el = document.getElementById(id);
        if (el) el.disabled = disabled;
    });
}

function disableBGMInputs(disabled) {
    var ids = ['bgm_genre', 'bgm_mood', 'bgm_instrument', 'bgm_theme', 'bgm_max_decode_steps', 'bgm_duration'];
    ids.forEach(function(id) {
        var el = document.getElementById(id);
        if (el) el.disabled = disabled;
    });
}

function disableTTAInputs(disabled) {
    document.getElementById('tta_text').disabled = disabled;
    var ids = ['tta_max_decode_steps', 'tta_cfg', 'tta_sigma', 'tta_temperature'];
    ids.forEach(function(id) {
        var el = document.getElementById(id);
        if (el) el.disabled = disabled;
    });
}

function showResult(id, success, message, audioUrl) {
    showResultCommon(id + '_result', success, message, audioUrl);
}

async function uploadAudioIfNeeded(inputId) {
    var input = document.getElementById(inputId);
    if (!input) return null;
    
    if (input.tagName === 'AUDIO-UPLOADER') {
        var file = input.getFile();
        if (file) {
            var formData = new FormData();
            formData.append('file', file);
            var resp = await fetch('/upload', { method: 'POST', body: formData });
            var data = await resp.json();
            if (data.success) {
                input.setAudioPath(data.filepath, null);
                return data.filepath;
            }
        }
        return input.getValue();
    }
    
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
    disableInstructInputs(true);
    
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
        disableInstructInputs(false);
        return;
    }

    var instructType = document.getElementById('instruct_type').value;
    var promptAudio = null;
    
    if (instructType !== 'ip' && instructType !== 'style') {
        promptAudio = await uploadAudioIfNeeded('instruct_audio_uploader');
        if (!promptAudio) {
            var uploader = document.getElementById('instruct_audio_uploader');
            promptAudio = uploader ? uploader.getValue() : null;
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
        ip = document.getElementById('instruct_ip_select').getValue();
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
    disableInstructInputs(false);
}

async function generateZeroShotTTS() {
    var btn = document.getElementById('zs_generate');
    btn.disabled = true;
    disableZeroShotInputs(true);
    
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
    var promptAudio = await uploadAudioIfNeeded('zs_audio_uploader');
    if (!promptAudio) {
        var uploader = document.getElementById('zs_audio_uploader');
        promptAudio = uploader ? uploader.getValue() : null;
    }
    var ip = document.getElementById('zs_ip_select').getValue();
    var promptText = document.getElementById('zs_prompt_text').value || null;

    if (!text) {
        clearInterval(timer);
        showResult('zs', false, '请输入文本');
        btn.disabled = false;
        disableZeroShotInputs(false);
        return;
    }

    if (!promptAudio && !ip) {
        clearInterval(timer);
        showResult('zs', false, '请上传参考音频或选择内置人物');
        btn.disabled = false;
        disableZeroShotInputs(false);
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
    disableZeroShotInputs(false);
}

async function generatePodcast() {
    var btn = document.getElementById('pod_generate');
    btn.disabled = true;
    disablePodcastInputs(true);
    
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
        disablePodcastInputs(false);
        return;
    }

    var promptAudios = [];
    for (var i = 1; i <= 3; i++) {
        var uploader = document.getElementById('pod_audio_' + i);
        var audio = await uploadAudioIfNeeded('pod_audio_' + i);
        if (!audio && uploader) {
            audio = uploader.getValue();
        }
        if (audio) {
            promptAudios.push(audio);
        }
    }

    if (promptAudios.length < 2) {
        clearInterval(timer);
        showResult('pod', false, '请至少上传两个说话人的参考音频');
        btn.disabled = false;
        disablePodcastInputs(false);
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
    disablePodcastInputs(false);
}

async function generateSpeechWithBGM() {
    var btn = document.getElementById('swb_generate');
    btn.disabled = true;
    disableSpeechWithBGMInputs(true);
    
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
        disableSpeechWithBGMInputs(false);
        return;
    }

    var promptAudio = await uploadAudioIfNeeded('swb_audio');
    if (!promptAudio) {
        var uploader = document.getElementById('swb_audio');
        promptAudio = uploader ? uploader.getValue() : null;
    }
    if (!promptAudio) {
        clearInterval(timer);
        showResult('swb', false, '请上传说话人参考音频');
        btn.disabled = false;
        disableSpeechWithBGMInputs(false);
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
    disableSpeechWithBGMInputs(false);
}

async function generateBGM() {
    var btn = document.querySelector('#bgm .generate-btn');
    btn.disabled = true;
    disableBGMInputs(true);
    
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

    btn.disabled = false;
    disableBGMInputs(false);
}

async function generateTTA() {
    var btn = document.querySelector('#tta .generate-btn');
    btn.disabled = true;
    disableTTAInputs(true);
    
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
        btn.disabled = false;
        disableTTAInputs(false);
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

    btn.disabled = false;
    disableTTAInputs(false);
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
