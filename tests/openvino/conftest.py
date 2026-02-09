import os
import tempfile
import pytest
from utils_tests import MODEL_NAMES


@pytest.fixture(scope="session", autouse=True)
def setup_tiny_cohere2_model():
    """Create a tiny Cohere2 model dynamically for testing."""
    try:
        from transformers import Cohere2Config, Cohere2ForCausalLM, AutoTokenizer
    except ImportError:
        # If Cohere2 is not available in transformers, skip this fixture
        yield
        return

    # Create temporary directory for the model
    tmpdir = tempfile.mkdtemp(prefix="tiny_cohere2_")
    
    # Get the original config to extract layer_types
    try:
        original_model_id = "estrogen/c4ai-command-r7b-12-2024"
        original_config = Cohere2Config.from_pretrained(original_model_id)
        
        # Extract the first two layer types from the original
        original_layer_types = getattr(original_config, "layer_types", None)
        
        if original_layer_types and len(original_layer_types) >= 2:
            # Keep only the first two types from the original layer_types
            layer_types = original_layer_types[:2]
        else:
            # Default to attention layers if layer_types is not present or too short
            # Using 2 attention layers ensures EXPECTED_NUM_SDPA = 2
            layer_types = ["attention", "attention"]
    except Exception:
        # If we can't load the original config, default to attention layers
        layer_types = ["attention", "attention"]
    
    # Create tiny config with specified parameters
    config = Cohere2Config(
        hidden_size=64,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        sliding_window=64,
        layer_types=layer_types,
        vocab_size=1000,  # Small vocab for testing
        max_position_embeddings=512,
    )
    
    # Create and save the model
    model = Cohere2ForCausalLM(config)
    model.save_pretrained(tmpdir)
    
    # Create and save a simple tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Use GPT2 tokenizer as base
        tokenizer.save_pretrained(tmpdir)
    except Exception:
        pass
    
    # Update MODEL_NAMES to point to the tiny model
    MODEL_NAMES["cohere2"] = tmpdir
    
    yield tmpdir
    
    # Cleanup
    import shutil
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
    except Exception:
        pass
