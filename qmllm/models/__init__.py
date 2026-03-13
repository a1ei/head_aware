import importlib
import os
import sys

import hf_transfer
from loguru import logger

from qmllm.models.internvl2 import InternVL2
from qmllm.models.llava_onevision import LLaVA_onevision
from qmllm.models.llava_v15 import LLaVA_v15
from qmllm.models.qwen2_vl import Qwen2_VL
from qmllm.models.vila import vila
try:
    from qmllm.models.qwen3_vl import Qwen3_VL
except Exception as e:
    print("Failed to import qwen3_vl; Please update it transformers to 5.0.0`")

from qmllm.utils.registry import MODEL_REGISTRY

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

logger.remove()
logger.add(sys.stdout, level="WARNING")

def get_process_model(model_name):
    return MODEL_REGISTRY[model_name]