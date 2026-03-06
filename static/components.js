class AudioUploader extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this._fileInput = null;
        this._audioUrl = null;
        this._audioPath = null;
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
        if (this._audioPath) return this._audioPath;
        
        var hiddenId = this.getAttribute('hidden-id');
        if (hiddenId) {
            var hiddenEl = document.getElementById(hiddenId);
            if (hiddenEl && hiddenEl.value) return hiddenEl.value;
        }
        
        return null;
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

        this._audioPath = null;

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

        this._audioPath = audioPath;

        var hiddenId = this.getAttribute('hidden-id');
        if (hiddenId) {
            var hiddenEl = document.getElementById(hiddenId);
            if (hiddenEl) hiddenEl.value = audioPath;
        }

        if (audioPath) {
            var url = configName ? '/config_audio/' + encodeURIComponent(configName) : audioPath;
            displayEl.innerHTML = '<div class="audio-container"><audio controls src="' + url + '"></audio><button class="clear-btn" id="clear-btn">✕</button></div><p style="font-size:12px;color:#666;">已加载参考音频</p>';
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
            displayEl.innerHTML = '<div class="audio-container"><audio controls src="' + url + '"></audio><button class="clear-btn" id="clear-btn">✕</button></div><p style="font-size:12px;color:#666;">' + file.name + '</p>';
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
                    width: 28px;
                    height: 28px;
                    padding: 0;
                    font-size: 16px;
                    line-height: 1;
                    background: #f44336;
                    color: white;
                    border: none;
                    border-radius: 50%;
                    cursor: pointer;
                    vertical-align: middle;
                    display: flex;
                    align-items: center;
                    justify-content: center;
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

class SearchSelect extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this._data = [];
        this._selectedValue = null;
    }

    static get observedAttributes() {
        return ['placeholder', 'hidden-id', 'data', 'display-key', 'value-key', 'filter-fields', 'label'];
    }

    connectedCallback() {
        this.render();
        this._initEventListeners();
    }

    attributeChangedCallback(name, oldValue, newValue) {
        if (oldValue === newValue) return;
        if (name === 'data') {
            try {
                this._data = JSON.parse(newValue) || [];
            } catch (e) {
                this._data = [];
            }
        } else if (name === 'display-key' || name === 'value-key' || name === 'filter-fields') {
            this.render();
        } else if (name === 'label') {
            this._updateLabel();
        }
    }

    get displayKey() {
        return this.getAttribute('display-key') || 'name';
    }

    get valueKey() {
        return this.getAttribute('value-key') || 'name';
    }

    get filterFields() {
        var fields = this.getAttribute('filter-fields') || 'name';
        return fields.split(',').map(function(f) { return f.trim(); });
    }

    getValue() {
        var hiddenInput = this.shadowRoot.getElementById('hidden-input');
        if (hiddenInput && hiddenInput.value) return hiddenInput.value;
        
        if (this._selectedValue) return this._selectedValue;
        
        return null;
    }

    setValue(value) {
        var self = this;
        var found = this._data.find(function(item) {
            return item[self.valueKey] === value;
        });
        
        if (found) {
            var input = this.shadowRoot.getElementById('search-input');
            var hiddenInput = this.shadowRoot.getElementById('hidden-input');
            
            this._selectedValue = value;
            if (input) input.value = found[this.displayKey];
            if (hiddenInput) hiddenInput.value = value;
            
            var hiddenId = this.getAttribute('hidden-id');
            if (hiddenId) {
                var hiddenEl = document.getElementById(hiddenId);
                if (hiddenEl) hiddenEl.value = value;
            }
        }
    }

    clear() {
        this._selectedValue = null;
        var input = this.shadowRoot.getElementById('search-input');
        var hiddenInput = this.shadowRoot.getElementById('hidden-input');
        var dropdown = this.shadowRoot.getElementById('dropdown');
        
        if (input) input.value = '';
        if (hiddenInput) hiddenInput.value = '';
        if (dropdown) dropdown.style.display = 'none';
        
        var hiddenId = this.getAttribute('hidden-id');
        if (hiddenId) {
            var hiddenEl = document.getElementById(hiddenId);
            if (hiddenEl) hiddenEl.value = '';
        }
    }

    setData(data) {
        this._data = data || [];
    }

    _updateLabel() {
        var label = this.shadowRoot.querySelector('label');
        if (label) label.textContent = this.getAttribute('label') || '';
    }

    _initEventListeners() {
        var self = this;
        var input = this.shadowRoot.getElementById('search-input');
        var dropdown = this.shadowRoot.getElementById('dropdown');

        if (!input || !dropdown) return;

        input.addEventListener('input', function() {
            self._showDropdown(this.value);
            if (!this.value) {
                self._selectedValue = null;
                var hiddenInput = self.shadowRoot.getElementById('hidden-input');
                if (hiddenInput) hiddenInput.value = '';
                
                var hiddenId = self.getAttribute('hidden-id');
                if (hiddenId) {
                    var hiddenEl = document.getElementById(hiddenId);
                    if (hiddenEl) hiddenEl.value = '';
                }
            }
        });

        input.addEventListener('focus', function() {
            self._showDropdown(self.shadowRoot.getElementById('search-input').value);
        });

        document.addEventListener('click', function(e) {
            if (!e.target.closest('search-select')) {
                dropdown.style.display = 'none';
            }
        });
    }

    _showDropdown(filter) {
        var input = this.shadowRoot.getElementById('search-input');
        var dropdown = this.shadowRoot.getElementById('dropdown');
        if (!input || !dropdown) return;

        dropdown.innerHTML = '';
        dropdown.style.display = 'block';

        var filterLower = (filter || '').toLowerCase().trim();
        var filterNoSpace = filterLower.replace(/\s+/g, '');

        var self = this;

        var matched = this._data.filter(function(item) {
            var searchText = self._getSearchText(item);
            return searchText.toLowerCase().includes(filterLower) ||
                   searchText.toLowerCase().replace(/\s+/g, '').includes(filterNoSpace);
        });

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
            div.addEventListener('click', function(e) {
                e.stopPropagation();
                self._selectItem(item);
            });
            dropdown.appendChild(div);
        });
    }

    _getSearchText(item) {
        var fields = this.filterFields;
        var texts = [];
        var self = this;
        fields.forEach(function(field) {
            var value = item[field];
            if (value) texts.push(value);
        });
        return texts.join(' ');
    }

    _selectItem(item) {
        var input = this.shadowRoot.getElementById('search-input');
        var hiddenInput = this.shadowRoot.getElementById('hidden-input');
        var dropdown = this.shadowRoot.getElementById('dropdown');

        this._selectedValue = item[this.valueKey];
        
        if (input) input.value = item[this.displayKey];
        if (hiddenInput) hiddenInput.value = item[this.valueKey];
        if (dropdown) dropdown.style.display = 'none';

        var hiddenId = this.getAttribute('hidden-id');
        if (hiddenId) {
            var hiddenEl = document.getElementById(hiddenId);
            if (hiddenEl) hiddenEl.value = item[this.valueKey];
        }
    }

    render() {
        var placeholder = this.getAttribute('placeholder') || '搜索...';
        var label = this.getAttribute('label') || '';
        var labelHtml = label ? '<label>' + label + '</label>' : '';

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
                .search-container {
                    position: relative;
                }
                input {
                    width: 100%;
                    padding: 10px 12px;
                    border: 1px solid #ddd;
                    border-radius: 6px;
                    font-size: 14px;
                    box-sizing: border-box;
                    transition: border-color 0.2s;
                }
                input:focus {
                    outline: none;
                    border-color: #4CAF50;
                }
                .dropdown {
                    position: absolute;
                    top: 100%;
                    left: 0;
                    right: 0;
                    max-height: 250px;
                    overflow-y: auto;
                    background: white;
                    border: 1px solid #ddd;
                    border-radius: 6px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                    z-index: 1000;
                    display: none;
                }
                .dropdown div {
                    padding: 10px 12px;
                    cursor: pointer;
                    transition: background 0.2s;
                }
                .dropdown div:hover {
                    background: #f5f5f5;
                }
                .dropdown .no-result {
                    color: #999;
                    cursor: default;
                }
                .dropdown .no-result:hover {
                    background: transparent;
                }
            </style>
            ${labelHtml}
            <div class="search-container">
                <input type="text" id="search-input" placeholder="${placeholder}" autocomplete="off">
                <input type="hidden" id="hidden-input">
                <div id="dropdown" class="dropdown"></div>
            </div>
        `;
    }
}

customElements.define('search-select', SearchSelect);
