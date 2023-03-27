#!/usr/bin/zsh
if [ 1 -ne $# ]; then
    echo "Misuse of submit script."
    exit 1
fi

echo "Use BLACK to format all Python files."
black ./src
echo "Finish formatting"

echo "Export conda environment"
conda env export > ./environment.yml
echo "Finish exporting"

echo "Push code to GitHub"
git add .
git commit -m $1
git push
echo "Script over"
