import logging

import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def load_model(model_id: str, device_type: str):
    """
    Responsible for loading tokenizer and model from huggingface
    :param model_id: Model needs to be loaded.
    :param device_type: "CUDA" if GPU is present else "CPU"
    :return: Loaded model and tokenizer
    """
    try:
        logger.info(f"model with model id {model_id} and device_type {device_type} ")
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_id)
        # loading model from huggingface
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_id,
                                                     torch_dtype=torch.float16,
                                                     low_cpu_mem_usage=True).to(device_type)
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to load model {model_id} on {device_type}: {e}")
        raise

def predict(base_prompt, llm_model, tokenizer):
    """
    Generate a response based on the user query.
    :param base_prompt: User Query
    :param llm_model: Huggingface model
    :param tokenizer: Tokenizer
    :return: Generated output
    """
    try:
        dialogue_template = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": base_prompt}
        ]
        generation_args = {
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.8,
            "do_sample": True,
        }
        from transformers import pipeline
        logger.info("Predicting output...")

        pipe = pipeline("text-generation", model=llm_model, tokenizer=tokenizer)
        output = pipe(dialogue_template, **generation_args)
        return output
    except Exception as e:
        logger.error(f"Failed to generate prediction: {e}")
        raise