#!/bin/bash

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
