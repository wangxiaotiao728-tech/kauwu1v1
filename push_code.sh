#!/usr/bin/env bash
set -e

# 默认提交信息
MSG="${1:-update code}"

echo "=============================="
echo "当前分支：$(git branch --show-current)"
echo "提交信息：$MSG"
echo "=============================="

# 检查是否有修改
if [ -z "$(git status --porcelain)" ]; then
    echo "没有检测到代码修改，无需提交。"
    exit 0
fi

echo "当前修改文件："
git status --short

echo "添加文件..."
git add .

echo "提交代码..."
git commit -m "$MSG"

BRANCH="$(git branch --show-current)"

echo "推送到 GitHub..."
if git rev-parse --abbrev-ref --symbolic-full-name @{u} >/dev/null 2>&1; then
    git push
else
    git push -u origin "$BRANCH"
fi

echo "完成：代码已推送到 GitHub。"
