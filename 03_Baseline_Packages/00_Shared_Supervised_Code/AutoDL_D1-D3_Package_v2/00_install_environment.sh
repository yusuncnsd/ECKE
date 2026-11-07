#!/bin/bash
set -euo pipefail
python -m pip install --upgrade pip
python -m pip install "transformers==4.55.4" "scikit-learn==1.7.2" "safetensors>=0.4" "sentencepiece>=0.2" "jieba==0.42.1" "tqdm>=4.66" "numpy<3"
python -c "import torch,transformers; print('torch=',torch.__version__,'cuda=',torch.cuda.is_available(),'gpu=',torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('transformers=',transformers.__version__)"
