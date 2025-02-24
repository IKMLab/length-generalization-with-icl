from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import MambaConfig, MambaPreTrainedModel, MambaModel
from transformers.models.mamba.modeling_mamba import MambaCache, MambaCausalLMOutput


class MambaForInContextLearning(MambaPreTrainedModel):

    def __init__(self, config: MambaConfig):
        super().__init__(config)

        self.input = nn.Linear(
            config.n_in_dims,
            config.hidden_size,
            bias=False,
        )
        self.backbone = MambaModel(config)
        self.output = nn.Linear(config.hidden_size, 1, bias=False)

        self.name = "MambaForInContextLearning"

        self.post_init()

    @staticmethod
    def _combine(xs_b, ys_b):
        """Interleaves the x's and the y's into a single sequence."""
        bsize, points, dim = xs_b.shape
        ys_b_wide = torch.cat(
            (
                ys_b.view(bsize, points, 1),
                torch.zeros(bsize, points, dim - 1, device=ys_b.device),
            ),
            axis=2,
        )
        zs = torch.stack((xs_b, ys_b_wide), dim=2)
        zs = zs.view(bsize, 2 * points, dim)

        return zs

    def forward(
        self,
        xs: torch.FloatTensor,
        ys: torch.FloatTensor,
        loss_func: torch.nn.Module,
        ids: Optional[torch.LongTensor] = None,
        cache_params: Optional[MambaCache] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, MambaCausalLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if ids is None:
            ids = torch.arange(ys.shape[1])
        else:
            ids = torch.tensor(ids)
            if max(ids) >= ys.shape[1] or min(ids) < 0:
                raise ValueError("idx contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)

        embeds = self.input(zs)
        mamba_outputs = self.backbone(
            inputs_embeds=embeds,
            cache_params=cache_params,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=use_cache,
        )
        hidden_state = mamba_outputs[0]
        prediction = self.output(hidden_state)

        loss = None
        loss = loss_func(prediction[:, ::2, 0][:, ids], ys[:, ids])

        if not return_dict:
            output = prediction[:, ::2, 0][:, ids]
            return ((loss,) + output) if loss is not None else output

        return MambaCausalLMOutput(
            loss=loss,
            logits=prediction[:, ::2, 0][:, ids],
            cache_params=mamba_outputs.cache_params,
            hidden_states=mamba_outputs.hidden_states,
        )
