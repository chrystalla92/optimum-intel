# Tiny Cohere Command-R Model for Testing

## Overview

The Cohere tests in this directory use a dynamically generated tiny version of the Command-R model (based on `estrogen/c4ai-command-r7b-12-2024`) rather than downloading a pre-existing model from HuggingFace.

## Implementation

The tiny model is created in `utils_tests.py` with the following characteristics:

### Model Configuration
- **Architecture**: Cohere Command-R
- **Vocab Size**: 256,128 (standard Cohere vocab)
- **Hidden Size**: 128 (reduced from full model)
- **Intermediate Size**: 512 (reduced)
- **Layers**: 2 (minimal for testing)
- **Attention Heads**: 4
- **Key-Value Heads**: 2 (grouped query attention)
- **Max Position Embeddings**: 512 (reduced context length)

### How It Works

1. **Creation**: The `create_tiny_cohere_command_r_model()` function generates a minimal Cohere model with random weights
2. **Initialization**: Model is created once at module import time via `get_tiny_cohere_model_path()`
3. **Storage**: Saved to a temporary directory that persists for the test session
4. **Usage**: `MODEL_NAMES["cohere"]` points to the dynamically created model path

### Tokenizer

The implementation attempts to use a compatible tokenizer with multiple fallback strategies:
1. First, tries to use the real Cohere tokenizer from `CohereForAI/c4ai-command-r-v01`
2. Falls back to GPT-2 tokenizer if Cohere tokenizer is unavailable
3. Creates a basic BPE tokenizer as a last resort

## Benefits

- **Fast Testing**: Tiny dimensions enable quick test execution
- **No External Dependencies**: No need to download models from HuggingFace during tests
- **Consistent Architecture**: Based on the actual Command-R architecture
- **Flexible**: Can be easily modified for different test requirements

## Modification

To change the model configuration, edit the `CohereConfig` parameters in the `create_tiny_cohere_command_r_model()` function in `utils_tests.py`.
