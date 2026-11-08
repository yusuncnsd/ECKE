#!/usr/bin/env bash
set -euo pipefail
python -m pip install -U "transformers==4.45.2" "huggingface-hub==0.26.5" "accelerate==1.1.1" "bitsandbytes==0.46.1" "sentencepiece==0.2.0" "safetensors>=0.4.5"
python - <<'PY'
import torch, transformers, bitsandbytes
print('torch=', torch.__version__, 'cuda=', torch.cuda.is_available())
print('transformers=', transformers.__version__, 'bitsandbytes=', bitsandbytes.__version__)
PY
