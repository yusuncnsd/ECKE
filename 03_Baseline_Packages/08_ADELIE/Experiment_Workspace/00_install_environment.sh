#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# AutoDL: Python 3.10/3.11、预装CUDA版PyTorch；不覆盖平台PyTorch。
python -m pip install "transformers==4.45.2" "huggingface-hub==0.26.5" "accelerate==1.1.1" "sentencepiece==0.2.0" "protobuf>=3.20,<6" "safetensors>=0.4.5"
python -c "import torch,transformers; print('torch=',torch.__version__,'CUDA=',torch.cuda.is_available(),'transformers=',transformers.__version__)"
