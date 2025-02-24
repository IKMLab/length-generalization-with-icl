def rope_scaling_validation(self):
    """Validate the `rope_scaling` configuration."""
    if self.rope_scaling is None:
        return

    if not isinstance(self.rope_scaling, dict):
        raise ValueError("`rope_scaling` must be a dictionary, "
                         f"got {self.rope_scaling}")
    rope_scaling_type = self.rope_scaling.get("type", None)
    rope_scaling_factor = self.rope_scaling.get("factor", None)
    if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic", "yarn", "dynamic-yarn"]:
        raise ValueError(
            f"`rope_scaling`'s name field must be one of ['linear', 'dynamic', 'yarn', 'dynamic-yarn'], got {rope_scaling_type}"
        )
    if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
        raise ValueError(f"`rope_scaling`'s factor field must be an float > 1, got {rope_scaling_factor}")
