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
