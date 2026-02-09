import os
import tempfile
from pathlib import Path

import pytest


def _create_tiny_cohere2_model():
    """Create a tiny Cohere2 model dynamically for testing."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    
    # Create config based on estrogen/c4ai-command-r7b-12-2024 but with tiny dimensions
    config = AutoConfig.from_pretrained("estrogen/c4ai-command-r7b-12-2024", trust_remote_code=True)
    
    # Make it tiny with specified parameters
    config.hidden_size = 64
    config.intermediate_size = 256
    config.num_hidden_layers = 2
    config.num_attention_heads = 8
    config.num_key_value_heads = 4
    config.sliding_window = 64
    
    # Keep only the first two types of the original layer types
    if hasattr(config, "layer_types") and config.layer_types:
        # Get unique layer types while preserving order
        unique_types = []
        for lt in config.layer_types:
            if lt not in unique_types:
                unique_types.append(lt)
        
        # Keep only first two unique types
        first_two_types = unique_types[:2]
        
        # Create layer_types list with only these two types for num_hidden_layers
        # This will give us 2 layers total
        config.layer_types = [first_two_types[i % len(first_two_types)] for i in range(config.num_hidden_layers)]
    
    # Create model from config
    model = AutoModelForCausalLM.from_config(config)
    
    # Create a temporary directory to save the model
    tmp_dir = tempfile.mkdtemp(prefix="tiny_cohere2_")
    model_path = Path(tmp_dir)
    
    # Save model and config
    model.save_pretrained(model_path)
    
    # Also save a tokenizer (use the original model's tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("estrogen/c4ai-command-r7b-12-2024", trust_remote_code=True)
    tokenizer.save_pretrained(model_path)
    
    return str(model_path)


# Global variable to cache the model path
_TINY_COHERE2_MODEL_PATH = None


def get_tiny_cohere2_model_path():
    """Get or create the tiny Cohere2 model path."""
    global _TINY_COHERE2_MODEL_PATH
    
    if _TINY_COHERE2_MODEL_PATH is None:
        _TINY_COHERE2_MODEL_PATH = _create_tiny_cohere2_model()
    
    return _TINY_COHERE2_MODEL_PATH
