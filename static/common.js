function showResultCommon(resultId, success, message, audioUrl) {
    var resultDiv = document.getElementById(resultId);
    if (!resultDiv) {
        resultDiv = document.getElementById('result');
    }
    if (success) {
        resultDiv.className = 'result success';
        resultDiv.style.display = 'block';
        resultDiv.innerHTML = '<p>' + message + '</p><audio controls src="' + audioUrl + '"></audio>';
    } else {
        resultDiv.className = 'result error';
        resultDiv.style.display = 'block';
        resultDiv.innerHTML = '<p>' + message + '</p>';
    }
}

function DropdownSearch(config) {
    this.inputId = config.inputId;
    this.dropdownId = config.dropdownId;
    this.hiddenId = config.hiddenId;
    this.data = config.data || [];
    this.displayKey = config.displayKey || 'name';
    this.valueKey = config.valueKey || 'name';
    this.filterFields = config.filterFields || ['name'];
    this.onSelect = config.onSelect || null;
    this.init();
}

DropdownSearch.prototype.init = function() {
    var self = this;
    var input = document.getElementById(this.inputId);
    if (!input) return;

    input.addEventListener('input', function() {
        self.show(this.value);
    });

    input.addEventListener('focus', function() {
        self.show(self.input.value);
    });
};

DropdownSearch.prototype.setData = function(data) {
    this.data = data;
};

DropdownSearch.prototype.show = function(filter) {
    var dropdown = document.getElementById(this.dropdownId);
    var input = document.getElementById(this.inputId);
    if (!dropdown || !input) return;

    dropdown.innerHTML = '';
    dropdown.style.display = 'block';

    var filterLower = (filter || '').toLowerCase().trim();
    var filterNoSpace = filterLower.replace(/\s+/g, '');

    var matched = this.data.filter(function(item) {
        var searchText = self.getSearchText(item);
        return searchText.toLowerCase().includes(filterLower) ||
               searchText.toLowerCase().replace(/\s+/g, '').includes(filterNoSpace);
    });

    var self = this;

    if (matched.length === 0) {
        var noResult = document.createElement('div');
        noResult.className = 'no-result';
        noResult.textContent = '无匹配结果';
        dropdown.appendChild(noResult);
        return;
    }

    matched.forEach(function(item) {
        var div = document.createElement('div');
        div.textContent = item[self.displayKey];
        div.onclick = function(e) {
            e.stopPropagation();
            input.value = item[self.displayKey];
            dropdown.style.display = 'none';
            if (self.hiddenId) {
                document.getElementById(self.hiddenId).value = item[self.valueKey];
            }
            if (self.onSelect) {
                self.onSelect(item);
            }
        };
        dropdown.appendChild(div);
    });
};

DropdownSearch.prototype.getSearchText = function(item) {
    var fields = this.filterFields;
    var texts = [];
    fields.forEach(function(field) {
        var value = item[field];
        if (value) texts.push(value);
    });
    return texts.join(' ');
};

DropdownSearch.prototype.hide = function() {
    var dropdown = document.getElementById(this.dropdownId);
    if (dropdown) dropdown.style.display = 'none';
};

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

function initSliderListeners() {
    document.querySelectorAll('input[type="range"]').forEach(function(slider) {
        slider.addEventListener('input', function() {
            var span = document.getElementById(this.id + '_val');
            if (span) span.textContent = this.value;
        });
    });
}

function showToast(message, isError) {
    var container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.style.cssText = 'position:fixed;top:20px;right:20px;z-index:9999;';
        document.body.appendChild(container);
    }
    var toast = document.createElement('div');
    toast.className = 'toast ' + (isError ? 'error' : 'success');
    toast.style.cssText = 'background:' + (isError ? '#f44336' : '#4CAF50') + ';color:white;padding:12px 20px;margin-bottom:10px;border-radius:4px;box-shadow:0 2px8px rgba(0,0,0,0.2);transition:all 0.3s ease-out;';
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(function() {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        setTimeout(function() {
            if (container.contains(toast)) {
                container.removeChild(toast);
            }
        }, 300);
    }, 3000);
}

class AudioUploader extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this._fileInput = null;
        this._audioUrl = null;
    }

    static get observedAttributes() {
        return ['label', 'hidden-id', 'prompt-text-id', 'accept', 'required'];
    }

    connectedCallback() {
        this.render();
        this._initEventListeners();
    }

    attributeChangedCallback(name, oldValue, newValue) {
        if (oldValue === newValue) return;
        if (name === 'label') this._updateLabel();
        if (name === 'required') this._updateRequired();
    }

    getValue() {
        return this._fileInput ? this._fileInput.dataset.filepath : null;
    }

    getFile() {
        return this._fileInput && this._fileInput.files ? this._fileInput.files[0] : null;
    }

    clear() {
        var displayEl = this.shadowRoot.getElementById('audio-display');
        var dropZone = this.shadowRoot.getElementById('drop-zone');
        if (displayEl) displayEl.innerHTML = '';
        if (dropZone) dropZone.style.display = 'flex';

        if (this._fileInput) {
            this._fileInput.value = '';
            delete this._fileInput.dataset.filepath;
        }

        var hiddenId = this.getAttribute('hidden-id');
        if (hiddenId) {
            var hiddenEl = document.getElementById(hiddenId);
            if (hiddenEl) hiddenEl.value = '';
        }

        var promptTextId = this.getAttribute('prompt-text-id');
        if (promptTextId) {
            var promptTextEl = document.getElementById(promptTextId);
            if (promptTextEl) promptTextEl.value = '';
        }
    }

    setAudioPath(audioPath, configName) {
        var self = this;
        var displayEl = this.shadowRoot.getElementById('audio-display');
        var dropZone = this.shadowRoot.getElementById('drop-zone');
        if (!displayEl || !dropZone) return;

        if (audioPath) {
            var url = configName ? '/config_audio/' + encodeURIComponent(configName) : audioPath;
            displayEl.innerHTML = '<div class="audio-container"><audio controls src="' + url + '"></audio><button class="clear-btn" id="clear-btn">清除</button></div><p style="font-size:12px;color:#666;">已加载参考音频</p>';
            dropZone.style.display = 'none';
            
            var clearBtn = this.shadowRoot.getElementById('clear-btn');
            if (clearBtn) {
                clearBtn.addEventListener('click', function(e) {
                    e.stopPropagation();
                    self.clear();
                });
            }
        } else {
            this.clear();
        }
    }

    _updateLabel() {
        var label = this.shadowRoot.querySelector('label');
        if (label) label.textContent = this.getAttribute('label') || '参考音频：';
    }

    _updateRequired() {
        var required = this.hasAttribute('required');
        if (this._fileInput) {
            this._fileInput.required = required;
        }
    }

    _initEventListeners() {
        var self = this;
        var dropZone = this.shadowRoot.getElementById('drop-zone');
        this._fileInput = this.shadowRoot.getElementById('file-input');

        if (!dropZone || !this._fileInput) return;

        dropZone.addEventListener('click', function() {
            self._fileInput.click();
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
                self._fileInput.files = files;
                self._handleFileSelect();
            }
        });

        this._fileInput.addEventListener('change', function() {
            self._handleFileSelect();
        });
    }

    _handleFileSelect() {
        var self = this;
        var fileInput = this._fileInput;
        var displayEl = this.shadowRoot.getElementById('audio-display');
        var dropZone = this.shadowRoot.getElementById('drop-zone');

        if (!fileInput || !displayEl || !dropZone) return;

        if (fileInput.files && fileInput.files[0]) {
            var file = fileInput.files[0];
            var url = URL.createObjectURL(file);
            displayEl.innerHTML = '<div class="audio-container"><audio controls src="' + url + '"></audio><button class="clear-btn" id="clear-btn">清除</button></div><p style="font-size:12px;color:#666;">' + file.name + '</p>';
            dropZone.style.display = 'none';
            
            var clearBtn = this.shadowRoot.getElementById('clear-btn');
            if (clearBtn) {
                clearBtn.addEventListener('click', function(e) {
                    e.stopPropagation();
                    self.clear();
                });
            }
        }
    }

    render() {
        var label = this.getAttribute('label') || '参考音频：';
        var accept = this.getAttribute('accept') || 'audio/*';
        var required = this.hasAttribute('required') ? 'required' : '';

        this.shadowRoot.innerHTML = `
            <style>
                :host {
                    display: block;
                    margin-bottom: 15px;
                }
                label {
                    display: block;
                    margin-bottom: 8px;
                    font-weight: 500;
                    color: #333;
                }
                .drop-zone {
                    border: 2px dashed #ccc;
                    border-radius: 8px;
                    padding: 30px;
                    text-align: center;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    background: #fafafa;
                }
                .drop-zone:hover {
                    border-color: #4CAF50;
                    background: #f0f8f0;
                }
                .drop-zone.dragover {
                    border-color: #4CAF50;
                    background: #e8f5e9;
                }
                .drop-zone p {
                    margin: 0;
                    color: #666;
                    font-size: 14px;
                }
                .audio-display {
                    margin-top: 10px;
                }
                .audio-display audio {
                    width: 100%;
                    max-width: 400px;
                }
                .audio-display p {
                    margin: 5px 0 0 0;
                    font-size: 12px;
                    color: #666;
                }
                .clear-btn {
                    margin-left: 10px;
                    padding: 6px 12px;
                    font-size: 12px;
                    background: #f44336;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    vertical-align: middle;
                    white-space: nowrap;
                }
                .clear-btn:hover {
                    background: #d32f2f;
                }
                .audio-container {
                    display: flex;
                    align-items: center;
                }
                .audio-container audio {
                    flex: 1;
                    max-width: calc(100% - 60px);
                }
                input[type="file"] {
                    display: none;
                }
            </style>
            <label>${label}</label>
            <div class="drop-zone" id="drop-zone">
                <p>拖拽音频文件到此处 或 点击选择</p>
            </div>
            <div class="audio-display" id="audio-display"></div>
            <input type="file" id="file-input" accept="${accept}" ${required}>
        `;
    }
}

customElements.define('audio-uploader', AudioUploader);
