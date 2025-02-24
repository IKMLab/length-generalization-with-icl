from .alibi import apply_alibi_patch
from .fire import apply_fire_patch
from .longlora import apply_longlora_attn_patch
from .nope import apply_nope_patch
from .resonance import apply_resonance_rope_patch
from .selfextend import apply_selfextend_attn_patch
from .yarn import apply_yarn_rope_patch


class Patch:

    def __init__(self, patch_name: str = None):
        self.patch = self._get_patch(patch_name)

    def _get_patch(self, patch_name: str = None):
        return {
            "alibi": apply_alibi_patch,
            "fire": apply_fire_patch,
            "longlora": apply_longlora_attn_patch,
            "nope": apply_nope_patch,
            "resonance": apply_resonance_rope_patch,
            "selfextend": apply_selfextend_attn_patch,
            "yarn": apply_yarn_rope_patch,
        }[patch_name] if patch_name else lambda: None

    def apply(self):
        self.patch()
