# AI 论文处理工具

这是一个本地运行的论文处理工具，支持命令行和 Streamlit Web App，用于处理英文科研论文 PDF，并生成：

- 中文翻译版 Markdown，图片以内嵌 Base64 写入
- 中文论文总结 Markdown
- 可选中文 HTML
- 可选结构化 JSON
- 可选 Excel 表格，包括 `paper_info`、`device_performance`、`stability_tests`、`characterization_methods`、`key_innovations`、`figures`

项目优先面向材料、器件、光伏和钙钛矿太阳能电池论文。图像裁切、器件性能抽取和稳定性抽取采用“LLM + 本地启发式 fallback”的工程方案，便于后续迭代。

## 安装

建议使用 Python 3.10 或更高版本。

```bash
cd paper_ai_tool
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
cd paper_ai_tool
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 配置 OpenAI API

复制环境变量模板：

```bash
cp .env.example .env
```

编辑 `.env`：

```bash
OPENAI_API_KEY=你的 API key
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=
```

如果使用 OpenAI-compatible 接口，例如 DeepSeek，可以这样配置：

```bash
OPENAI_API_KEY=你的 DeepSeek API key
OPENAI_MODEL=deepseek-v4-pro
OPENAI_BASE_URL=https://api.deepseek.com
```

如果不设置 `OPENAI_API_KEY`，程序不会崩溃，会使用降级方案：

- 正文保留英文，并对内置术语做少量替换
- 总结基于规则生成粗略版本
- 器件性能和稳定性测试用正则尽量抽取

## 运行

### 命令行运行

命令行模式不需要打开 VSCode 或浏览器：

```bash
python cli.py "/path/to/paper.pdf"
```

也可以不带 PDF 参数启动，程序会提示你拖入文件：

```bash
python cli.py
```

提示出现后，把 PDF 从访达或 Windows 文件管理器拖到终端窗口，然后按回车即可。

### 双击运行

macOS：

双击项目里的 `run_paper_tool.command`，打开终端窗口后拖入 PDF，再按回车。

Windows：

双击项目里的 `run_paper_tool.bat`，打开命令行窗口后拖入 PDF，再按回车。

这两个脚本会自动优先使用项目根目录的 `.venv`，也兼容 `paper_ai_tool/.venv`。

如果 Windows 双击后窗口仍然立刻关闭，通常是脚本没有在标准 `cmd.exe` 中运行。可以右键 `run_paper_tool.bat` 选择“在终端中打开”，或先打开 PowerShell / CMD 后执行：

```bat
cd /d C:\path\to\paper_ai_tool
run_paper_tool.bat
```

新版 `run_paper_tool.bat` 会在启动时打印脚本目录、虚拟环境 Python 路径，并在错误时停留在窗口中，便于定位问题。

指定输出目录：

```bash
python cli.py "/path/to/paper.pdf" -o outputs
```

如果不指定 `-o`，默认输出到当前用户桌面：

- macOS：`~/Desktop`
- Windows：当前用户的 `Desktop`

需要固定输出到其他目录时，可以在 `.env` 设置：

```bash
OUTPUT_DIR=/path/to/output
```

常用参数：

```bash
python cli.py "/path/to/paper.pdf" \
  --image-format png \
  --no-compress-images
```

### Web App 运行

```bash
streamlit run app.py
```

浏览器打开 Streamlit 页面后：

1. 上传 PDF
2. 选择输出目录
3. 选择图片格式，默认 PNG
4. 选择是否压缩图片
5. 点击“开始处理”

## 输出开关

`.env` 中控制实际生成哪些文件。默认只输出中文 Markdown 和总结 Markdown：

```bash
OUTPUT_MARKDOWN=1
OUTPUT_SUMMARY=1
OUTPUT_HTML=0
OUTPUT_JSON=0
OUTPUT_EXCEL=0
OUTPUT_SOURCE_PDF=0
OUTPUT_DEBUG_FIGURES=0
RUN_STRUCTURED_EXTRACTION=0
```

需要 HTML 时：

```bash
OUTPUT_HTML=1
```

需要结构化 JSON 或 Excel 时：

```bash
OUTPUT_JSON=1
OUTPUT_EXCEL=1
```

`OUTPUT_JSON=1` 或 `OUTPUT_EXCEL=1` 时会自动开启结构化抽取。若只想生成总结但也希望总结参考结构化抽取结果，可以单独设置：

```bash
RUN_STRUCTURED_EXTRACTION=1
```

默认关闭结构化抽取可以减少一次大模型长文本调用，速度更快，也更省 token。

默认不复制原 PDF、不输出调试图片。需要保留时：

```bash
OUTPUT_SOURCE_PDF=1
OUTPUT_DEBUG_FIGURES=1
```

## 输出文件

假设输入为 `paper.pdf`，输出目录为 `outputs/paper/`：

```text
outputs/paper/
  paper_translated_embedded.md
  paper_summary.md
  paper_processing.log
```

`paper_translated_embedded.md` 中的图片使用 Base64 内嵌，不依赖 `debug_figures/`。

如果开启全部可选输出：

```text
outputs/paper/
  paper_translated_embedded.md
  paper_translated.html
  paper_summary.md
  paper_structured.json
  paper_tables.xlsx
  paper_processing.log
```

当前版本已移除 PDF 导出，不再依赖 WeasyPrint 或 Playwright。

## 技术实现

- PDF 读取与页面渲染：PyMuPDF
- 图像裁切：按图注位置推断区域，渲染页面截图后裁切
- 文本抽取：PyMuPDF 文本块
- LLM：OpenAI API，封装在 `src/llm_client.py`
- Markdown/HTML：Python markdown
- Excel：pandas + openpyxl

## 结构化数据策略

`src/structured_extractor.py` 会优先调用 LLM 输出 JSON，字段包含：

- `paper_info`
- `sections`
- `figures`
- `device_performance`
- `stability_tests`
- `experimental_methods`
- `characterization_methods`
- `key_innovations`
- `new_knowledge`
- `limitations`
- `warnings`

如果 LLM 不可用或 JSON 无法解析，会启用规则抽取，尽量从正文、图注和表格文本中识别：

- PCE、certified PCE、steady-state PCE
- VOC、JSC、FF、active area
- MPP、damp heat、thermal、light/dark cycling、storage 等稳定性测试
- retained PCE、duration、cycles、temperature、humidity

不确定字段保持 `null` 或空字符串，不编造数据。每条性能和稳定性记录尽量保留 `source_text`、`source_page`、`source_figure`。

## 后续优化接口

当前实现保留了可迭代接口：

- `FigureExtractor._infer_figure_bbox`：可替换为更强的版面识别或人工校正
- `StructuredExtractor._heuristic_extract`：可继续扩展正则和表格解析
- `LLMClient.chat`：可替换为其他模型提供商
- `PDFParser._build_sections`：可接入更精细的章节识别
