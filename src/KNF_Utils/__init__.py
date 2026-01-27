# Import all node classes (NodeBypasser is frontend-only)
from .nodes import (
    KNF_Organizer,
    GeminiVideoCaptioner,
    NODE_CLASS_MAPPINGS as NODES_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as NODES_DISPLAY_NAME_MAPPINGS
)

# Import memory monitor nodes
from .memory_monitor import (
    NODE_CLASS_MAPPINGS as MEMORY_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as MEMORY_DISPLAY_NAME_MAPPINGS
)

# Import committed noise nodes (for chunk consistency)
from .committed_noise import (
    NODE_CLASS_MAPPINGS as COMMITTED_NOISE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as COMMITTED_NOISE_DISPLAY_NAME_MAPPINGS
)

# Import RoPE utility nodes (for temporal position alignment)
from .rope_utils import (
    NODE_CLASS_MAPPINGS as ROPE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as ROPE_DISPLAY_NAME_MAPPINGS
)

# Merge all node mappings
NODE_CLASS_MAPPINGS = {
    **NODES_CLASS_MAPPINGS,
    **MEMORY_CLASS_MAPPINGS,
    **COMMITTED_NOISE_CLASS_MAPPINGS,
    **ROPE_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **NODES_DISPLAY_NAME_MAPPINGS,
    **MEMORY_DISPLAY_NAME_MAPPINGS,
    **COMMITTED_NOISE_DISPLAY_NAME_MAPPINGS,
    **ROPE_DISPLAY_NAME_MAPPINGS,
}

# Export the mappings for ComfyUI
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']