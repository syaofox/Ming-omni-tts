---
name: ming-omni-tts-frontend
description: Ming-omni-tts 前端开发规范。用于修改 WebUI/API 模板、CSS 样式、JavaScript 代码时使用。包括模板路径规范、CSS 样式规范、JS 模块化规范。当用户提及修改前端、模板、CSS、JS、WebUI、API 界面等相关任务时，必须使用此 skill。
---

# Ming-omni-tts 前端开发规范

## 路径规范

### 容器 vs 本地路径
- **容器内**：`/app/`（项目根目录）
- **本地**：`/mnt/github/Ming-omni-tts/`

### Python 脚本路径
使用相对路径获取项目根目录：
```python
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建路径：os.path.join(current_dir, "saved_configs", "httpTts.json")
```

### 适用文件
- `templates/webui.html`
- `templates/api.html`
- `static/webui.css`
- `static/app.css`
- `static/*.js`

---

## CSS 规范

### 禁止内联样式
**禁止**在 HTML 中使用 `style="..."` 属性。所有样式必须写在 CSS 文件中。

### 公共样式类
使用现有 CSS 类：

| 类名 | 用途 |
|------|------|
| `.form-group` | 表单项容器 |
| `.form-row` | 表单行（flex 布局） |
| `.form-row-inline` | 同行表单（IP + 端口） |
| `.params-grid` | 参数网格（2列） |
| `.section-title` | 区块标题（h3） |
| `.row.compact` | 紧凑行 |

### 新增样式
如需新增样式，添加到 `static/app.css` 或 `static/webui.css`：
```css
.form-row-inline {
    display: flex;
    flex-wrap: wrap;
    gap: 15px;
    align-items: flex-end;
}

.form-row-inline .form-group {
    flex: 1;
    min-width: 120px;
}

.form-row-inline .form-group:first-child {
    flex: 2;
}
```

---

## 模板规范

### 标题
使用 `<h3 class="section-title">` 而非 `<h1>` 或 `<h2>`：
```html
<h3 class="section-title">生成 HTTP TTS 配置</h3>
```

### 同行表单
使用 `form-row-inline` 类：
```html
<div class="form-row-inline">
    <div class="form-group">
        <label>服务器 IP：</label>
        <input type="text" id="config_ip" value="{{ server_ip }}">
    </div>
    <div class="form-group">
        <label>端口：</label>
        <input type="number" id="config_port" value="7861">
    </div>
</div>
```

### 输入框
使用标准 HTML5 类型：
- 文本：`input type="text"`
- 数字：`input type="number"`

---

## JS 模块化规范

### 现有模块
- `static/utils.js` - 通用工具函数
- `static/components.js` - 组件定义
- `static/webui.js` - WebUI 逻辑

### 新增 JS
按功能模块化拆分到 `static/` 目录，避免在 HTML 中内联 JS 代码。

### 引用方式
```html
<script src="{{ url_for('static', filename='utils.js') }}"></script>
<script src="{{ url_for('static', filename='components.js') }}"></script>
<script src="{{ url_for('static', filename='webui.js') }}"></script>
```

---

## 常见错误避免

### 1. 路径错误
错误示例：
```python
output_path = "/mnt/github/Ming-omni-tts/saved_configs/httpTts.json"
```

正确示例：
```python
current_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(current_dir, "saved_configs", "httpTts.json")
```

### 2. 内联样式
错误示例：
```html
<div class="form-group" style="flex: 2;">
    <input type="text" style="width: 100%;">
</div>
```

正确示例：
```html
<!-- 在 CSS 中定义 -->
<div class="form-row-inline">
    <div class="form-group">
        <input type="text">
    </div>
</div>
```

### 3. 标题层级
错误示例：
```html
<h2>生成 HTTP TTS 配置</h2>
```

正确示例：
```html
<h3 class="section-title">生成 HTTP TTS 配置</h3>
```

---

## 参考资料

### CSS 文件位置
- `static/app.css` - 通用样式
- `static/webui.css` - WebUI 样式

### 模板文件位置
- `templates/webui.html` - WebUI 界面
- `templates/api.html` - API 界面
