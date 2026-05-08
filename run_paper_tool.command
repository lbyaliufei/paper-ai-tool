#!/bin/zsh
set -e

SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR"

if [ -x "../.venv/bin/python" ]; then
  PYTHON="../.venv/bin/python"
elif [ -x ".venv/bin/python" ]; then
  PYTHON=".venv/bin/python"
else
  echo "未找到虚拟环境 Python。"
  echo "请先在项目根目录执行：python3 -m venv .venv && source .venv/bin/activate && pip install -r paper_ai_tool/requirements.txt"
  echo ""
  read "dummy?按回车退出..."
  exit 1
fi

echo "AI 论文处理工具"
echo "当前目录：$SCRIPT_DIR"
echo ""
"$PYTHON" cli.py

echo ""
read "dummy?处理结束。按回车关闭窗口..."
