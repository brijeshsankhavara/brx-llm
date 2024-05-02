from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .config import BRXConfig
from .model import BRXForCausalLM, BRXModel

# Lets AutoConfig / AutoModelForCausalLM.from_pretrained load BRX checkpoints once `brx` is imported.
AutoConfig.register("brx", BRXConfig)
AutoModel.register(BRXConfig, BRXModel)
AutoModelForCausalLM.register(BRXConfig, BRXForCausalLM)

__all__ = ["BRXConfig", "BRXModel", "BRXForCausalLM"]
