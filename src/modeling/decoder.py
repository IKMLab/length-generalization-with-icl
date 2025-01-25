from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (
    GPT2Model,
    GPT2PreTrainedModel,
    GPT2ForTokenClassification,
    LlamaModel,
    LlamaPreTrainedModel,
)
from transformers.modeling_outputs import TokenClassifierOutput, CausalLMOutput


class GPT2ForTokenClassification(GPT2ForTokenClassification):

    def __combine_labels(self, inputs, labels):
        combined = torch.stack((inputs, labels), dim=2)
        combined = combined.view(inputs.shape[0], -1)

        return combined

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        combined: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, TokenClassifierOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids=combined,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0][:, ::2, :]
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        logits = logits

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


class GPT2ForInContextLearning(GPT2PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.input = nn.Linear(config.n_in_dims, config.n_embd, bias=False)
        self.transformer = GPT2Model(config)
        self.output = nn.Linear(config.n_embd, 1, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Model problem type
        # self.problem_type = config.problem_type

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
        loss_func: Optional[torch.nn.Module] = None,
        ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if ids is None:
            ids = torch.arange(ys.shape[1])
        else:
            ids = torch.tensor(ids)
            if max(ids) >= ys.shape[1] or min(ids) < 0:
                raise ValueError("idx contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)

        embeds = self.input(zs)
        transformer_outputs = self.transformer(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_state = transformer_outputs[0]
        prediction = self.output(hidden_state)

        loss = None
        loss = loss_func(prediction[:, ::2, 0][:, ids], ys[:, ids])

        if not return_dict:
            output = prediction[:, ::2, 0][:, ids]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=prediction[:, ::2, 0][:, ids],
            hidden_states=hidden_state,
            attentions=transformer_outputs.attentions,
        )


class LlamaForTokenizeICL(LlamaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.input = nn.Linear(
            config.n_in_dims,
            config.hidden_size,
            bias=False,
        )
        self.transformer = LlamaModel(config)
        self.output = nn.Linear(config.hidden_size, 1, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # self.problem_type = "regression"

        self.post_init()

    def forward(
        self,
        inputs: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        ids: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if ids is None:
            ids = torch.arange(labels.shape[1])
        else:
            ids = torch.tensor(ids)
            if max(ids) >= labels.shape[1] or min(ids) < 0:
                raise ValueError("idx contain indices where xs and ys are not defined")

        embeds = self.input(inputs)
        transformer_outputs = self.transformer(inputs_embeds=embeds)
        hidden_state = transformer_outputs[0]
        prediction = self.output(hidden_state)

        loss = None
        if labels is not None:
            labels = labels.to(prediction.device)
            loss_fct = nn.MSELoss()
            loss = loss_fct(prediction[:, 2::4, 0][:, ids], labels[:, ids])

        if not return_dict:
            output = prediction[:, 2::4, 0][:, ids]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=prediction[:, 2::4, 0][:, ids],
            hidden_states=hidden_state,
            attentions=transformer_outputs.attentions,
        )


class LlamaForInContextLearning(LlamaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.input = nn.Linear(
            config.n_in_dims,
            config.hidden_size,
            bias=False,
        )
        self.transformer = LlamaModel(config)
        self.output = nn.Linear(config.hidden_size, 1, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Model problem type
        # self.problem_type = config.problem_type
        self.name = "LlamaForInContextLearning"

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
        loss_func: Optional[torch.nn.Module] = None,
        ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutput]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if ids is None:
            ids = torch.arange(ys.shape[1])
        else:
            ids = torch.tensor(ids)
            if max(ids) >= ys.shape[1] or min(ids) < 0:
                raise ValueError("idx contain indices where xs and ys are not defined")
        zs = self._combine(xs, ys)

        embeds = self.input(zs)
        transformer_outputs = self.transformer(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_state = transformer_outputs[0]
        prediction = self.output(hidden_state)

        loss = None
        loss = loss_func(prediction[:, ::2, 0][:, ids], ys[:, ids])

        if not return_dict:
            output = prediction[:, ::2, 0][:, ids]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=prediction[:, ::2, 0][:, ids],
            hidden_states=hidden_state,
            attentions=transformer_outputs.attentions,
        )
