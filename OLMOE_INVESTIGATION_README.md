# OlMOE MoE Architecture Investigation

This directory contains investigation artifacts for adding OlMOE model support to optimum-intel OpenVINO export.

## Purpose

Investigate the internal MoE (Mixture of Experts) architecture of the OlMOE model to determine the correct patching strategy for OpenVINO export. The model uses dynamic control flow (loops over experts) that breaks `torch.jit.trace`, requiring specialized patching.

## Files

### 1. `investigate_olmoe.py`
Python script that loads and analyzes the OlMOE model structure.

**Usage:**
```bash
python investigate_olmoe.py
```

**What it does:**
- Loads `allenai/OLMoE-1B-7B-0924` with `trust_remote_code=True`
- Analyzes model structure (layers, MoE modules, experts)
- Examines router mechanism and expert organization
- Identifies control flow patterns that break torch.jit.trace
- Compares structure with existing Afmoe implementation
- Prints detailed analysis to console

**Requirements:**
- `transformers >= 4.45.0`
- `torch`
- Internet connection (for model download)

### 2. `OLMOE_FINDINGS.md`
Comprehensive documentation template for investigation results.

**Sections:**
- Architecture Analysis (experts, router, shared experts)
- Control Flow Patterns (torch.jit.trace issues)
- Comparison with Existing MoE Implementations (Afmoe, Qwen2MoE)
- Patching Strategy Recommendations (with code templates)
- Implementation Checklist
- Next Steps

**Usage:**
- Run `investigate_olmoe.py` to get actual model structure
- Fill in the template sections with findings
- Use as reference for implementing the patcher

## Investigation Workflow

```
┌─────────────────────────────┐
│  1. Run Investigation       │
│     python investigate_     │
│     olmoe.py                │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  2. Review Output           │
│     - Expert structure      │
│     - Router interface      │
│     - Control flow issues   │
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  3. Document Findings       │
│     Update OLMOE_FINDINGS.md│
└──────────┬──────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  4. Implement Patcher       │
│     Add to model_patcher.py │
└─────────────────────────────┘
```

## Background: Why Patching is Needed

### The Problem

MoE models typically use loops to iterate over experts:

```python
for expert_idx in range(num_experts):
    if token_uses_expert[expert_idx]:
        output += expert[expert_idx](hidden_states) * routing_weight[expert_idx]
```

This breaks `torch.jit.trace` because:
1. Loop iterations are dynamic (depends on expert selection)
2. Conditional logic based on tensor values
3. Cannot be statically traced

### The Solution: Vectorization

Replace loops with batched matrix multiplications:

```python
# Concatenate all expert weights
all_gate_weights = torch.concat([expert[i].gate.weight for i in range(num_experts)])

# Process all experts at once with torch.bmm
all_outputs = torch.bmm(hidden_states, all_gate_weights)

# Apply routing weights and sum
final_output = (all_outputs * routing_weights).sum(dim=0)
```

Benefits:
- ✅ No loops or conditionals
- ✅ torch.jit.trace compatible
- ✅ OpenVINO can optimize batched operations
- ✅ Often faster due to vectorization

## Reference Implementations

### Afmoe (Fully Vectorized)
**Location:** `optimum/exporters/openvino/model_patcher.py:7443-7521`

**Strategy:**
- Concatenates all expert weights into 3D tensors
- Uses `torch.bmm` for all expert computations
- Completely eliminates loops

**Use case:** Best for uniform expert structures, standard FFN experts

### Qwen2MoE (Loop with index_add)
**Location:** `optimum/exporters/openvino/model_patcher.py:5105-5148`

**Strategy:**
- Still uses loop over experts
- Uses `index_add_` for accumulation
- Has shared expert with gating

**Use case:** When full vectorization is complex due to expert selection patterns

## Expected OlMOE Architecture

Based on the paper (https://arxiv.org/abs/2409.02060):

```
OLMoEForCausalLM
└── model (OLMoEModel)
    └── layers (ModuleList)
        └── OLMoEDecoderLayer
            ├── self_attn
            └── mlp (OLMoESparseMoeBlock)  ← MoE component
                ├── gate (router)
                ├── experts (ModuleList)
                │   └── Expert[i]
                │       ├── gate_proj
                │       ├── up_proj
                │       └── down_proj
                └── shared_expert (optional)
```

**Key attributes to verify:**
- [ ] `layer.mlp` or `layer.block_sparse_moe` or `layer.moe`
- [ ] `moe.experts` ModuleList
- [ ] `expert.gate_proj`, `expert.up_proj`, `expert.down_proj`
- [ ] `moe.gate` or `moe.router`
- [ ] `moe.shared_expert` or `moe.shared_experts`

## Next Actions

After completing this investigation:

1. **Update model_patcher.py**
   - Add `olmoe_moe_forward_patched()` function
   - Add `OlmoeModelPatcher` class
   
2. **Update model_configs.py**
   - Add `OlmoeOpenVINOConfig` class
   - Register in tasks_manager

3. **Add test support**
   - Create tiny-random-olmoe model
   - Add to test suites

4. **Validate**
   - Test export with tiny model
   - Test with real OlMOE-1B-7B-0924
   - Verify output correctness

## Questions to Answer

The investigation script should answer:

- [x] What is the exact path to the MoE module? (`layer.mlp`, `layer.moe`, etc.)
- [x] What are the expert attribute names? (`gate_proj`, `w1`, etc.)
- [x] What is the router interface? (inputs/outputs)
- [x] Are there shared experts? How are they structured?
- [x] What control flow patterns need patching? (specific loops/conditionals)
- [x] Does the structure match Afmoe closely enough to use similar vectorization?

## Troubleshooting

### Model Loading Issues

**Error:** `trust_remote_code not enabled`
```python
# Solution: Add trust_remote_code=True
model = AutoModelForCausalLM.from_pretrained(
    "allenai/OLMoE-1B-7B-0924",
    trust_remote_code=True
)
```

**Error:** `Out of memory`
```python
# Solution: Use low_cpu_mem_usage and smaller dtype
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,  # or bfloat16
    device_map="cpu",
    low_cpu_mem_usage=True
)
```

### Investigation Script Issues

**Error:** `Cannot find source code`
- This is expected for `trust_remote_code` models
- The script will still analyze the structure
- Check model files at `~/.cache/huggingface/hub/models--allenai--OLMoE-1B-7B-0924`

## Additional Resources

- **OlMOE Paper**: https://arxiv.org/abs/2409.02060
- **OlMOE Model Card**: https://huggingface.co/allenai/OLMoE-1B-7B-0924
- **OpenVINO Export Docs**: https://huggingface.co/docs/optimum-intel/en/openvino/export
- **torch.jit.trace Docs**: https://pytorch.org/docs/stable/generated/torch.jit.trace.html
