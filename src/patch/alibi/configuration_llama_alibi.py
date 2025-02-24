def rope_scaling_validation(self):
    """Validate the `rope_scaling` configuration."""
    if self.rope_scaling is None:
        return

    if not isinstance(self.rope_scaling, dict):
        raise ValueError("`rope_scaling` must be a dictionary, "
                         f"got {self.rope_scaling}")

    rope_scaling_type = self.rope_scaling.get("type", None)
    if rope_scaling_type is None or rope_scaling_type not in ["alibi"]:
        raise ValueError(f"`rope_scaling`'s name field must be one of ['alibi'], got {rope_scaling_type}")
