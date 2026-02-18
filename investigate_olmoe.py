#!/usr/bin/env python3
"""
Investigation script for OlMOE MoE architecture.

This script loads the OlMOE model and analyzes its internal MoE structure
to determine the correct patching strategy for OpenVINO export.
"""

import sys
import inspect
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def analyze_model_structure(model):
    """Analyze the basic structure of the OlMOE model."""
    print_section("MODEL STRUCTURE")
    
    print("Model type:", type(model).__name__)
    print("Model config type:", type(model.config).__name__)
    print("\nTop-level attributes:")
    for attr in dir(model):
        if not attr.startswith('_') and not callable(getattr(model, attr)):
            print(f"  - {attr}")
    
    print("\nModel layers accessible via:")
    if hasattr(model, 'model'):
        print("  model.model.layers")
        print(f"  Number of layers: {len(model.model.layers)}")
    elif hasattr(model, 'transformer'):
        print("  model.transformer.layers")
        print(f"  Number of layers: {len(model.transformer.layers)}")
    else:
        print("  Unknown layer structure")


def analyze_layer_structure(layer, layer_idx=0):
    """Analyze the structure of a single layer."""
    print_section(f"LAYER {layer_idx} STRUCTURE")
    
    print("Layer type:", type(layer).__name__)
    print("\nLayer attributes:")
    for attr in sorted(dir(layer)):
        if not attr.startswith('_'):
            obj = getattr(layer, attr)
            if not callable(obj):
                print(f"  - {attr}: {type(obj).__name__}")
    
    # Check for MoE-related attributes
    print("\nMoE-related attributes:")
    moe_attrs = ['mlp', 'block_sparse_moe', 'moe', 'feed_forward', 'experts']
    for attr in moe_attrs:
        if hasattr(layer, attr):
            obj = getattr(layer, attr)
            print(f"  ✓ {attr}: {type(obj).__name__}")
        else:
            print(f"  ✗ {attr}: not found")


def analyze_moe_module(moe_module):
    """Analyze the MoE module structure in detail."""
    print_section("MOE MODULE DETAILED ANALYSIS")
    
    print("MoE Module type:", type(moe_module).__name__)
    
    # Print all attributes
    print("\nMoE Module attributes:")
    for attr in sorted(dir(moe_module)):
        if not attr.startswith('_'):
            obj = getattr(moe_module, attr)
            if not callable(obj):
                print(f"  - {attr}: {type(obj).__name__}")
    
    # Check for expert organization
    print("\n" + "-" * 40)
    print("EXPERT ORGANIZATION")
    print("-" * 40)
    if hasattr(moe_module, 'experts'):
        experts = moe_module.experts
        print(f"✓ Experts found: {type(experts).__name__}")
        if isinstance(experts, torch.nn.ModuleList):
            print(f"  Number of experts: {len(experts)}")
            if len(experts) > 0:
                print(f"  Expert type: {type(experts[0]).__name__}")
                print(f"\n  First expert attributes:")
                for attr in sorted(dir(experts[0])):
                    if not attr.startswith('_'):
                        obj = getattr(experts[0], attr)
                        if not callable(obj):
                            print(f"    - {attr}: {type(obj).__name__}")
    else:
        print("✗ No 'experts' attribute found")
    
    # Check for shared experts
    print("\n" + "-" * 40)
    print("SHARED EXPERTS")
    print("-" * 40)
    if hasattr(moe_module, 'shared_experts'):
        print(f"✓ Shared experts found: {type(moe_module.shared_experts).__name__}")
    elif hasattr(moe_module, 'shared_expert'):
        print(f"✓ Shared expert found: {type(moe_module.shared_expert).__name__}")
    else:
        print("✗ No shared experts found")
    
    # Check for router
    print("\n" + "-" * 40)
    print("ROUTER MECHANISM")
    print("-" * 40)
    router_attrs = ['router', 'gate', 'gating', 'gating_network']
    router_found = False
    for attr in router_attrs:
        if hasattr(moe_module, attr):
            obj = getattr(moe_module, attr)
            print(f"✓ Router found as '{attr}': {type(obj).__name__}")
            print(f"  Router attributes:")
            for sub_attr in sorted(dir(obj)):
                if not sub_attr.startswith('_'):
                    sub_obj = getattr(obj, sub_attr)
                    if not callable(sub_obj):
                        print(f"    - {sub_attr}: {type(sub_obj).__name__}")
            router_found = True
            break
    
    if not router_found:
        print("✗ No router found")
    
    # Check config
    print("\n" + "-" * 40)
    print("MOE CONFIGURATION")
    print("-" * 40)
    if hasattr(moe_module, 'config'):
        config = moe_module.config
        print("Config attributes:")
        for attr in sorted(dir(config)):
            if not attr.startswith('_') and 'expert' in attr.lower() or 'moe' in attr.lower() or 'router' in attr.lower() or 'top' in attr.lower():
                try:
                    val = getattr(config, attr)
                    if not callable(val):
                        print(f"  - {attr}: {val}")
                except:
                    pass


def analyze_forward_method(moe_module):
    """Analyze the forward method for control flow patterns."""
    print_section("FORWARD METHOD ANALYSIS")
    
    print("Analyzing forward method for control flow patterns...\n")
    
    # Get the forward method source code
    try:
        forward_method = moe_module.forward
        source = inspect.getsource(forward_method)
        
        print("Forward method source code:")
        print("-" * 80)
        print(source)
        print("-" * 80)
        
        # Check for problematic patterns
        print("\n" + "-" * 40)
        print("CONTROL FLOW PATTERNS (potential torch.jit.trace issues)")
        print("-" * 40)
        
        patterns = {
            'for': 'Loop detected',
            'while': 'While loop detected',
            'if ': 'Conditional detected',
            '.item()': 'Tensor to Python conversion detected',
            'range': 'Range iteration detected',
            'enumerate': 'Enumerate iteration detected',
        }
        
        for pattern, description in patterns.items():
            if pattern in source:
                print(f"  ⚠ {description}")
                # Find and print the lines
                lines = source.split('\n')
                for i, line in enumerate(lines):
                    if pattern in line:
                        print(f"     Line {i}: {line.strip()}")
        
    except Exception as e:
        print(f"Could not retrieve source code: {e}")
        print("This is expected if the module is from trust_remote_code.")


def compare_with_afmoe(moe_module):
    """Compare OlMOE structure with Afmoe reference."""
    print_section("COMPARISON WITH AFMOE")
    
    print("Checking for Afmoe-like attributes:\n")
    
    afmoe_patterns = {
        'experts': 'Array of expert modules',
        'shared_experts': 'Shared expert module',
        'router': 'Router mechanism',
        'expert_bias': 'Expert bias term',
        'config.num_experts': 'Number of experts config',
    }
    
    for pattern, description in afmoe_patterns.items():
        if '.' in pattern:
            # Handle nested attributes
            parts = pattern.split('.')
            obj = moe_module
            found = True
            for part in parts:
                if hasattr(obj, part):
                    obj = getattr(obj, part)
                else:
                    found = False
                    break
            if found:
                print(f"  ✓ {pattern}: {description} (FOUND)")
            else:
                print(f"  ✗ {pattern}: {description} (NOT FOUND)")
        else:
            if hasattr(moe_module, pattern):
                print(f"  ✓ {pattern}: {description} (FOUND)")
            else:
                print(f"  ✗ {pattern}: {description} (NOT FOUND)")


def main():
    """Main investigation function."""
    print_section("OlMOE MoE ARCHITECTURE INVESTIGATION")
    
    model_name = "allenai/OLMoE-1B-7B-0924"
    print(f"Loading model: {model_name}")
    print("This may take a few minutes...\n")
    
    try:
        # Load model with trust_remote_code
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        
        print("✓ Model loaded successfully\n")
        
        # Analyze model structure
        analyze_model_structure(model)
        
        # Analyze first layer
        first_layer = model.model.layers[0]
        analyze_layer_structure(first_layer, 0)
        
        # Find and analyze MoE module
        moe_module = None
        if hasattr(first_layer, 'mlp'):
            moe_module = first_layer.mlp
        elif hasattr(first_layer, 'block_sparse_moe'):
            moe_module = first_layer.block_sparse_moe
        elif hasattr(first_layer, 'moe'):
            moe_module = first_layer.moe
        
        if moe_module is not None:
            analyze_moe_module(moe_module)
            analyze_forward_method(moe_module)
            compare_with_afmoe(moe_module)
        else:
            print("⚠ WARNING: Could not find MoE module in layer")
        
        # Summary
        print_section("INVESTIGATION SUMMARY")
        print("Analysis complete. Key findings:")
        print("1. Check the 'EXPERT ORGANIZATION' section for expert structure")
        print("2. Check the 'ROUTER MECHANISM' section for routing logic")
        print("3. Check the 'FORWARD METHOD ANALYSIS' section for control flow issues")
        print("4. Check the 'COMPARISON WITH AFMOE' section for similarity assessment")
        print("\nNext steps:")
        print("- Document the findings in OLMOE_FINDINGS.md")
        print("- Determine if vectorization approach (like Afmoe) is needed")
        print("- Design the patching strategy for model_patcher.py")
        
    except Exception as e:
        print(f"\n✗ Error loading or analyzing model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
