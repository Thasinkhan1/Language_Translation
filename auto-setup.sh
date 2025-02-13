#!/bin/bash

git init
git branch -m main

echo ".venv/" > .gitignore

echo "pip3 install -r requirements.txt --no-cache-dir" > lib-install.sh
chmod +x lib-install.sh

echo "pandas" > requirements.txt
echo "numpy" >> requirements.txt
echo "transformers" >> requirements.txt
echo "torch --index-url https://download.pytorch.org/whl/cpu" >> requirements.txt
echo "torchvision --index-url https://download.pytorch.org/whl/cpu" >> requirements.txt
echo "torchaudio --index-url https://download.pytorch.org/whl/cpu" >> requirements.txt
echo "datasets" >> requirements.txt
echo "indic-nlp-library" >> requirements.txt

python3 -m venv .venv
echo "source /home/aiml-machine/AiML-projects/language-translation/.venv/bin/activate" > activate-venv.sh
chmod +x activate-venv.sh
