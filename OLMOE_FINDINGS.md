# OlMOE MoE Architecture Investigation Findings

**Date**: Investigation Task  
**Model**: `allenai/OLMoE-1B-7B-0924`  
**Purpose**: Determine correct patching strategy for OpenVINO export

---

## Investigation Results

### Model Loading
- **Status**: To be executed
- **Command**: `python investigate_olmoe.py`
- **Requirements**: transformers >= 4.45.0, trust_remote_code=True

---

## Architecture Analysis

### 1. Expert Organization

**Location**: `model.layers[0].mlp` or similar  

**Structure**:
```
[To be filled after running investigation script]
```

**Key Attributes**:
- [ ] Expert array path (e.g., `.experts`)
- [ ] Number of experts
- [ ] Expert module type
- [ ] Individual expert components (gate_proj, up_proj, down_proj)

---

### 2. Shared Experts

**Status**: To be determined

**Findings**:
```
[To be filled after running investigation script]
```

**Questions**:
- [ ] Does OlMOE have shared experts?
- [ ] If yes, what is the attribute path?
- [ ] How are shared experts processed?

---

### 3. Router Mechanism

**Location**: To be determined

**Structure**:
```
[To be filled after running investigation script]
```

**Key Points**:
- [ ] Router attribute path
- [ ] Input/output of router
- [ ] Routing weight computation
- [ ] Expert selection logic

---

### 4. Control Flow Patterns

**Problematic patterns for torch.jit.trace**:

```python
[To be filled after analyzing forward method]
```

**Identified Issues**:
- [ ] For loops over experts
- [ ] Conditional statements based on tensor values
- [ ] Dynamic expert selection
- [ ] Tensor.item() calls

---

## Comparison with Existing MoE Implementations

### Strategy 1: Afmoe (Vectorized - lines 7443-7521)

**Afmoe Structure**:
```python
# Location: model.layers[i].mlp (when moe_enabled=True)

# Router
routing_weights, selected_experts = self.router(hidden_states, self.expert_bias)

# Experts
self.experts[i]  # ModuleList of experts
    .gate_proj.weight
    .up_proj.weight
    .down_proj.weight
    .act_fn

# Shared experts (optional)
self.shared_experts(hidden_states)

# Config
self.config.num_experts
```

**Afmoe Patching Strategy (FULLY VECTORIZED)**:
1. **Pre-processing**: Concatenate all expert weights into single 3D tensors:
   ```python
   gate_projs = torch.concat([experts[i].gate_proj.weight.unsqueeze(0) 
                               for i in range(num_experts)]).transpose(1,2).float()
   # Shape: [num_experts, hidden_dim, intermediate_dim]
   ```

2. **Vectorized computation**: Use batched matmul (torch.bmm) - NO LOOPS:
   ```python
   hidden_states = hidden_states.repeat(num_experts, 1)  # Replicate for all experts
   gate = torch.bmm(hidden_states, gate_projs)           # All experts at once
   up = torch.bmm(hidden_states, up_projs)
   gate_up = act_fn(gate) * up
   next_states = torch.bmm(gate_up, down_projs)
   ```

3. **Routing**: Apply routing weights as element-wise multiplication
4. **Aggregation**: Sum expert outputs across expert dimension

**Key Issue Solved**: Eliminates loops over experts, making it torch.jit.trace compatible

---

### Strategy 2: Qwen2MoE (Loop-based - lines 5105-5148)

**Qwen2MoE Structure**:
```python
# Location: Qwen2MoeSparseMoeBlock

# Router
router_logits = self.gate(hidden_states)
routing_weights, selected_experts = torch.topk(routing_weights, self.top_k)

# Experts
self.experts[expert_idx]  # ModuleList

# Shared expert
self.shared_expert(hidden_states)
self.shared_expert_gate(hidden_states)  # Gating mechanism
```

**Qwen2MoE Patching Strategy (LOOP WITH INDEX_ADD)**:
1. **Router**: Compute routing weights and select top-k experts per token
2. **Expert mask**: One-hot encode selected experts
3. **Loop over experts** (STILL HAS LOOP):
   ```python
   for expert_idx in range(self.num_experts):
       expert_layer = self.experts[expert_idx]
       idx, top_x = torch.where(expert_mask[expert_idx])
       current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
       current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
       final_hidden_states.index_add_(0, top_x, current_hidden_states)
   ```
4. **Shared expert**: Separate computation with sigmoid gating

**Key Issue**: Still contains loop, but uses index_add_ for accumulation

---

### OlMOE vs Existing Implementations

**Critical Questions to Answer**:
1. Does OlMOE have a similar expert structure (gate_proj, up_proj, down_proj)?
2. Does the original OlMOE forward method use loops over experts?
3. Does OlMOE have shared experts?
4. What is the router output format?
5. Is the expert structure uniform (all experts have same layers)?

**Expected Similarities with Afmoe**:
- Likely has ModuleList of expert FFN modules
- Probably uses gate_proj, up_proj, down_proj pattern (common in Llama-style models)
- May have shared experts (mentioned in OlMOE paper)
- Routing mechanism with top-k selection

**Potential Differences**:
- OlMOE may have its own routing strategy
- May use different activation functions
- Possible different expert gating mechanism
- Could have unique shared expert implementation

---

## MoE Patching Decision Tree

Based on investigation results, choose appropriate strategy:

### Choose Afmoe-style Vectorization IF:
- ✓ Original forward has explicit loop: `for expert_idx in range(num_experts):`
- ✓ All experts have uniform structure (same layers)
- ✓ Expert computation is: FFN with gate/up/down projections
- ✓ No complex control flow within expert selection

### Choose Qwen2MoE-style Patching IF:
- ✓ Top-k sparse routing per token
- ✓ Dynamic expert selection makes vectorization complex
- ✓ Expert gating is token-dependent

### Choose Minimal/Custom Patching IF:
- ✓ Forward method already uses vectorized operations
- ✓ No explicit loops over experts
- ✓ Only needs dtype fixes or minor adjustments

---

## Patching Strategy Recommendations

### Recommended Approach: Afmoe-style Vectorization (PREFERRED)

**Rationale**:
- OlMOE is a Llama-based MoE model (similar architecture to Afmoe/Arcee Trinity)
- Standard MoE implementations typically use loops over experts
- Vectorization with torch.bmm is proven to work for similar architectures
- Based on OlMOE paper, it follows standard MoE patterns

**Implementation Plan**:

```python
def olmoe_moe_forward_patched(self, hidden_states):
    """
    Patched forward for OlMOE MoE layer.
    Vectorizes expert computation using batched matmul.
    """
    num_experts = self.config.num_experts  # or self.num_experts
    batch_size, seq_len, hidden_dim = hidden_states.shape
    
    # Router computation (exact interface TBD)
    routing_weights, selected_experts = self.router(hidden_states)  
    
    # Scatter routing weights to dense matrix
    new_routing_weights = torch.zeros(
        batch_size * seq_len, num_experts, 
        dtype=routing_weights.dtype
    )
    new_routing_weights.scatter_(dim=1, index=selected_experts, src=routing_weights)
    
    hidden_states = hidden_states.view(-1, hidden_dim)
    
    # Handle shared experts (if present)
    if hasattr(self, 'shared_experts') and self.shared_experts is not None:
        shared_output = self.shared_experts(hidden_states)
    else:
        shared_output = torch.zeros_like(hidden_states)
    
    # Replicate hidden states for all experts
    hidden_states = hidden_states.repeat(num_experts, 1)
    hidden_states = hidden_states.view(num_experts, -1, hidden_dim)
    
    # Get activation function from first expert
    act_fn = self.experts[0].act_fn  # or ACT2FN[config.hidden_act]
    
    # Vectorized expert computation
    gate = torch.bmm(hidden_states, self.gate_projs)
    up = torch.bmm(hidden_states, self.up_projs)
    gate_up = act_fn(gate) * up
    next_states = torch.bmm(gate_up, self.down_projs)
    
    # Apply routing weights
    next_states = next_states.view(num_experts, batch_size, -1, hidden_dim)
    next_states = next_states * new_routing_weights.transpose(0, 1).view(
        num_experts, batch_size, -1
    )[..., None]
    next_states = next_states.sum(dim=0)
    
    # Add shared expert output
    shared_output = shared_output.view(batch_size, -1, hidden_dim)
    output = shared_output + next_states
    
    return output.view(batch_size, seq_len, hidden_dim)


class OlmoeModelPatcher(OVDecoderModelPatcher):
    def __enter__(self):
        super().__enter__()
        
        # Iterate through layers and patch MoE modules
        for idx, layer in enumerate(self._model.model.layers):
            # Identify MoE module (exact path TBD from investigation)
            # Could be: layer.mlp, layer.block_sparse_moe, layer.moe, etc.
            olmoe_moe = layer.mlp  # or appropriate attribute
            
            if olmoe_moe is not None:  # Check if this layer has MoE
                num_experts = olmoe_moe.config.num_experts
                
                # Save original forward
                olmoe_moe._orig_forward = olmoe_moe.forward
                
                # Replace with patched forward
                olmoe_moe.forward = types.MethodType(
                    olmoe_moe_forward_patched, olmoe_moe
                )
                
                # Concatenate expert weights for vectorization
                # Convert to float32 to match hidden_states dtype
                olmoe_moe.gate_projs = (
                    torch.concat(
                        tuple(olmoe_moe.experts[i].gate_proj.weight.unsqueeze(0) 
                              for i in range(num_experts)),
                        dim=0,
                    )
                    .transpose(1, 2)
                    .float()
                )
                
                olmoe_moe.up_projs = (
                    torch.concat(
                        tuple(olmoe_moe.experts[i].up_proj.weight.unsqueeze(0) 
                              for i in range(num_experts)),
                        dim=0,
                    )
                    .transpose(1, 2)
                    .float()
                )
                
                olmoe_moe.down_projs = (
                    torch.concat(
                        tuple(olmoe_moe.experts[i].down_proj.weight.unsqueeze(0) 
                              for i in range(num_experts)),
                        dim=0,
                    )
                    .transpose(1, 2)
                    .float()
                )
    
    def __exit__(self, exc_type, exc_value, traceback):
        super().__exit__(exc_type, exc_value, traceback)
        
        for idx, layer in enumerate(self._model.model.layers):
            olmoe_moe = layer.mlp  # or appropriate attribute
            
            if hasattr(olmoe_moe, '_orig_forward'):
                # Restore original forward
                olmoe_moe.forward = olmoe_moe._orig_forward
                
                # Clean up concatenated weight tensors
                del olmoe_moe.gate_projs
                del olmoe_moe.up_projs
                del olmoe_moe.down_projs
```

**Adjustments Needed After Investigation**:
1. Confirm attribute path to MoE module (`.mlp`, `.moe`, etc.)
2. Verify router interface and return format
3. Confirm expert structure (gate_proj, up_proj, down_proj names)
4. Verify shared expert handling
5. Check if all layers have MoE or only some
6. Confirm activation function access

---

## Implementation Checklist

Based on investigation results:

- [ ] Expert layer attribute paths documented
- [ ] Router mechanism understood
- [ ] Shared experts handling determined
- [ ] Control flow issues identified
- [ ] Vectorization need assessed
- [ ] Patching strategy decided
- [ ] Code changes planned for model_patcher.py

---

## How to Run the Investigation

### Prerequisites
```bash
pip install transformers>=4.45.0 torch
```

### Execute Investigation
```bash
# This will load OlMOE model and analyze its structure
python investigate_olmoe.py

# Expected runtime: 5-10 minutes (includes model download)
# Output: Detailed analysis printed to console
```

### What the Script Analyzes
1. Model top-level structure and layer organization
2. MoE module location and attributes
3. Expert array structure and individual expert components
4. Router mechanism and interface
5. Shared expert presence and implementation
6. Forward method source code and control flow patterns
7. Comparison with Afmoe structure

### Update This Document
After running the investigation, update the following sections:
- [ ] "Expert Organization" - fill in actual attribute paths
- [ ] "Shared Experts" - document findings
- [ ] "Router Mechanism" - document interface
- [ ] "Control Flow Patterns" - list identified issues
- [ ] "OlMOE vs Existing Implementations" - complete comparison
- [ ] "Required Changes" - adjust patching code based on findings

---

## Next Steps

### Phase 1: Complete Investigation (CURRENT)
1. ✅ Created investigation script (`investigate_olmoe.py`)
2. ✅ Analyzed existing MoE patchers (Afmoe, Qwen2MoE)
3. ✅ Designed patching strategy framework
4. ⏳ **TODO**: Run investigation script with actual OlMOE model
5. ⏳ **TODO**: Document specific findings in this file

### Phase 2: Implement Patcher
1. Add OlMOE patcher to `optimum/exporters/openvino/model_patcher.py`
2. Register OlMOE in model configs
3. Add tests for OlMOE export

### Phase 3: Create Tiny Model Helper
1. Create `tiny-random-olmoe` model creation function
2. Add to test utilities

### Phase 4: Validation
1. Test export with tiny model
2. Test export with real OlMOE-1B-7B-0924
3. Verify output correctness

---

## Additional Notes

### Key Insights from Existing Implementations

1. **torch.jit.trace incompatibility**: Loops and conditionals break tracing
   - Solution: Replace with vectorized operations using torch.bmm

2. **dtype consistency**: Hidden states are fp32 during tracing
   - Solution: Convert concatenated weights to .float()

3. **Expert weight concatenation**: Creates 3D tensors [num_experts, in_dim, out_dim]
   - Transpose(1,2) to get correct dimensions for bmm
   - unsqueeze(0) before concat to add expert dimension

4. **Routing weight handling**: Scatter sparse weights to dense matrix
   - Enables element-wise multiplication with all expert outputs

5. **Shared experts**: Computed separately and added to final output
   - May have gating mechanism (see Qwen2MoE)

### OlMOE Model Information

**Model**: `allenai/OLMoE-1B-7B-0924`
- **Architecture**: MoE-based decoder model
- **Base**: Similar to Llama/Mistral architecture with MoE layers
- **Experts**: 7B total parameters with sparse activation (1B active)
- **Paper**: https://arxiv.org/abs/2409.02060

**Expected Structure** (based on paper):
- 16 transformer layers
- Each layer has MoE FFN
- 64 experts per layer  
- Top-8 routing (8 experts active per token)
- Shared expert for common computation
- Grouped query attention

### Resources
- OlMOE Paper: https://arxiv.org/abs/2409.02060
- Afmoe Implementation: `model_patcher.py:7443-7521`
- Qwen2MoE Implementation: `model_patcher.py:5105-5148`
- HuggingFace Model: https://huggingface.co/allenai/OLMoE-1B-7B-0924
