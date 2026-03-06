---
name: frontend-development
description: Ming-omni-tts 项目前端开发规范。用于修改 WebUI/API 模板、CSS、JS 时触发。包含路径规范、CSS 模块化、禁止内联样式等规则。
---

## 路径规范

- 容器内路径：`/app/`
- 本地路径：`/mnt/github/Ming-omni-tts/`
- Python 脚本使用相对路径：`os.path.dirname(os.path.abspath(__file__))`

## CSS 模块化

| 模板文件 | 样式文件 |
|---------|---------|
| templates/webui.html | static/webui.css |
| templates/api.html | static/api.css |
| 公共组件 | static/app.css |

## 禁止内联样式

**禁止**在 HTML 中使用 `style="..."`，必须使用 CSS 类。

### 常用布局类
- `.form-row` - 表单行（flex 布局）
- `.form-row-inline` - 同行输入框（IP + 端口）
- `.params-grid` - 参数网格（2列）
- `.section-title` - 区块标题（h3）

### 常用样式类
- `.form-group` - 表单组
- `.row` / `.col` - 行列布局
- `.slider-group` - 滑块组

## JS 模块化

按功能拆分到 `static/` 目录：
- static/utils.js - 工具函数
- static/components.js - 组件
- static/webui.js - WebUI 逻辑
- static/api.js - API 逻辑

## 适用文件

- templates/webui.html
- templates/api.html
- static/app.css
- static/webui.css
- static/api.css
- static/*.js
