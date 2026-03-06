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
            audioDisplay.innerHTML = '<audio controls src="' + url + '"></audio><p class="audio-display-text">' + file.name + '</p>';
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
        document.body.appendChild(container);
    }
    var toast = document.createElement('div');
    toast.className = 'toast ' + (isError ? 'error' : 'success');
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
