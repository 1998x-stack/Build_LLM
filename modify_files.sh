#!/bin/bash

# ⚠️  一次性仓库清理生成器  /  ONE-TIME REPO MAINTENANCE GENERATOR
# 仅用于创建仓库时处理文本格式；不要重新运行（DO NOT re-run）：
# 它会删除 LaTeX 风格标记（\( \) \[ \]）并移除所有 【...】标注，
# 且作用于全部被跟踪文本文件（排除 .git）。
# 如确需再次运行，请设置环境变量 BUILD_LLM_GEN=1。

# Run-guard: refuse to re-run unless explicitly allowed.
if [ "${BUILD_LLM_GEN:-}" != "1" ]; then
    echo "[modify_files.sh] 这是一次性脚手架/格式化脚本，已冻结，默认不重新运行。"
    echo "[modify_files.sh] This one-time formatting generator is frozen; do NOT re-run."
    echo "[modify_files.sh] It would strip LaTeX-style markers and remove 【...】 annotations across tracked files."
    echo "[modify_files.sh] To force re-run, set BUILD_LLM_GEN=1."
    exit 0
fi

# 设置语言环境以避免非法字节序列错误
export LC_CTYPE=C

# 获取当前脚本所在的目录
DIRECTORY="$(cd "$(dirname "$0")" && pwd)"
echo "Current directory: $DIRECTORY"

# 查找目录中的所有文件并进行替换和删除操作，排除 .git 文件夹
find "$DIRECTORY" -type f ! -name "modify_files.sh" ! -path "*/.git/*" -exec sed -i '' '
    s/\\(/$/g; 
    s/\\)/$/g; 
    s/\\\[/\$\$/g; 
    s/\\\]/\$\$/g;
    s/【.*】//g
' {} +

echo "All files have been processed."
