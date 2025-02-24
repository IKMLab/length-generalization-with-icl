from .resonance_rope import (
    ResonanceRotaryEmbedding,
    ResonanceNTKScalingRotaryEmbedding,
    ResonanceYaRNScalingRotaryEmbedding,
)


def init_rope(self):
    if self.config.rope_scaling is None:
        self.rotary_emb = ResonanceRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
    else:
        scaling_type = self.config.rope_scaling["type"]
        scaling_factor = self.config.rope_scaling["factor"]
        if scaling_type == "ntk":
            self.rotary_emb = ResonanceNTKScalingRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.rope_theta,
            )
        elif scaling_type == "yarn":
            self.rotary_emb = ResonanceYaRNScalingRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                scaling_factor=scaling_factor,
                base=self.rope_theta,
            )
        else:
            raise ValueError(f"Unknown Resonance RoPE scaling type {scaling_type}")
