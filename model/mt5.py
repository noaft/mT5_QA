from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# in my code only for using mt5
class model:
    """
    Init model and device using.
    model_name: is name model it will be call transformer or path of model if have model in local
    tokenizer: tokenizer split and embbeding text to token -> vector embeeding
    device: device using in your machine
    """
    def __init__(model_name, tokenizer, device, path = None):
        # if want to call transformer download or call api for using model
        if path is None:
            model_name = model_name
            tokenizer = tokenizer