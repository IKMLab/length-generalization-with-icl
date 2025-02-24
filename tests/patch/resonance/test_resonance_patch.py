import unittest

from parameterized import parameterized
from transformers import LlamaConfig, is_torch_available, logging, set_seed
from transformers.testing_utils import (
    require_torch,
    torch_device,
)

if is_torch_available():
    import torch

    from transformers import LlamaModel

    from src.patch.resonance import (
        ResonanceRotaryEmbedding,
        ResonanceLinearScalingRotaryEmbedding,
        ResonanceNTKScalingRotaryEmbedding,
        ResonanceYaRNScalingRotaryEmbedding,
    )
    from src.patch.resonance.original import (
        OriginalResonanceRotaryEmbedding,
        OriginalResonanceLinearScaledRotaryEmbedding,
        OriginalResonanceNTKScaledRotaryEmbedding,
        OriginalResonanceYaRNScaledRotaryEmbedding,
    )
    from ...test_configuration_common import ConfigTester
    from ...test_modeling_common import ids_tensor


class LlamaModelTester:

    def __init__(
        self,
        parent,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="silu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=32,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        pad_token_id=0,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.pad_token_id = pad_token_id
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_tensor([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = torch.tril(torch.ones(self.batch_size, self.seq_length)).to(torch_device)

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_tensor([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_tensor([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_tensor([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_tensor([self.batch_size], self.num_choices)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return LlamaConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


@require_torch
class TestLlamaYaRNModel(unittest.TestCase):

    def setUp(self):
        self.model_tester = LlamaModelTester(self)
        self.config_tester = ConfigTester(self, config_class=LlamaConfig, hidden_size=37)

    @parameterized.expand([("linear",), ("dynamic",), ("yarn"), ("dynamic-yarn")])
    def test_model_rope_scaling_from_config(self, scaling_type):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        short_input = ids_tensor([1, 10], config.vocab_size)
        long_input = ids_tensor([1, int(config.max_position_embeddings * 8)], config.vocab_size)

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        original_model = LlamaModel(config)
        original_model.to(torch_device)
        original_model.eval()
        original_short_output = original_model(short_input).last_hidden_state
        original_long_output = original_model(long_input).last_hidden_state

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        config.rope_scaling = {"type": scaling_type, "factor": 4.0}
        scaled_model = LlamaModel(config)
        scaled_model.to(torch_device)
        scaled_model.eval()
        scaled_short_output = scaled_model(short_input).last_hidden_state
        scaled_long_output = scaled_model(long_input).last_hidden_state

        # Dynamic scaling does not change the RoPE embeddings until it receives an input longer than the original
        # maximum sequence length, so the outputs for the short input should match.
        if scaling_type == "dynamic" or scaling_type == "dynamic-yarn":
            self.assertTrue(torch.allclose(original_short_output, scaled_short_output, atol=1e-5))
        else:
            self.assertFalse(torch.allclose(original_short_output, scaled_short_output, atol=1e-5))

        # The output should be different for long inputs
        self.assertFalse(torch.allclose(original_long_output, scaled_long_output, atol=1e-5))

    def test_model_rope_scaling(self):
        config, _ = self.model_tester.prepare_config_and_inputs_for_common()
        hidden_size = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = hidden_size // num_heads
        scaling_factor = 4
        short_input_length = 10
        long_input_length = int(config.max_position_embeddings * 8)

        # Inputs
        x = torch.randn(1, dtype=torch.float32, device=torch_device)  # used exlusively to get the dtype and the device
        position_ids_short = torch.arange(short_input_length, dtype=torch.long, device=torch_device)
        position_ids_short = position_ids_short.unsqueeze(0)
        position_ids_long = torch.arange(long_input_length, dtype=torch.long, device=torch_device)
        position_ids_long = position_ids_long.unsqueeze(0)

        # Original Resonance RoPE
        original_res_rope = OriginalResonanceRotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        ).to(torch_device)
        original_res_cos_short, original_res_sin_short = original_res_rope(x, position_ids_short.shape[-1])
        original_res_cos_long, original_res_sin_long = original_res_rope(x, position_ids_long.shape[-1])
        torch.testing.assert_close(original_res_cos_short, original_res_cos_long[:, :, :short_input_length, :])
        torch.testing.assert_close(original_res_sin_short, original_res_sin_long[:, :, :short_input_length, :])

        # Resonance RoPE
        res_rope = ResonanceRotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=scaling_factor,
        ).to(torch_device)
        res_cos_short, res_sin_short = res_rope(x, position_ids_short)
        res_cos_long, res_sin_long = res_rope(x, position_ids_long)
        torch.testing.assert_close(res_cos_short, res_cos_long[:, :short_input_length])
        torch.testing.assert_close(res_sin_short, res_sin_long[:, :short_input_length])

        torch.testing.assert_close(original_res_cos_short, res_cos_short.unsqueeze(0))
        torch.testing.assert_close(original_res_sin_short, res_sin_short.unsqueeze(0))
        torch.testing.assert_close(original_res_cos_long, res_cos_long.unsqueeze(0))
        torch.testing.assert_close(original_res_sin_long, res_sin_long.unsqueeze(0))

        # # Original Resonance linear RoPE
        # original_res_linear = OriginalResonanceLinearScaledRotaryEmbedding(
        #     head_dim,
        #     max_position_embeddings=config.max_position_embeddings,
        #     base=config.rope_theta,
        # ).to(torch_device)
        # original_res_linear_cos_short, original_res_linear_sin_short = original_res_linear(
        #     x, position_ids_short.shape[-1])
        # original_res_linear_cos_long, original_res_linear_sin_long = original_res_linear(x, position_ids_long.shape[-1])
        # torch.testing.assert_close(original_res_linear_cos_short,
        #                            original_res_linear_cos_long[:, :, :short_input_length, :])
        # torch.testing.assert_close(original_res_linear_sin_short,
        #                            original_res_linear_sin_long[:, :, :short_input_length, :])

        # # Resonance linear RoPE
        # res_linear = ResonanceLinearScaledRotaryEmbedding(
        #     head_dim,
        #     max_position_embeddings=config.max_position_embeddings,
        #     base=config.rope_theta,
        #     scaling_factor=scaling_factor,
        # ).to(torch_device)
        # res_linear_cos_short, res_linear_sin_short = res_linear(x, position_ids_short)
        # res_linear_cos_long, res_linear_sin_long = res_linear(x, position_ids_long)
        # torch.testing.assert_close(res_linear_cos_short, res_linear_cos_long[:, :short_input_length])
        # torch.testing.assert_close(res_linear_sin_short, res_linear_sin_long[:, :short_input_length])

        # torch.testing.assert_close(original_res_linear_cos_short, res_linear_cos_short.unsqueeze(0))
        # torch.testing.assert_close(original_res_linear_sin_short, res_linear_sin_short.unsqueeze(0))
        # torch.testing.assert_close(original_res_linear_cos_long, res_linear_cos_long.unsqueeze(0))
        # torch.testing.assert_close(original_res_linear_sin_long, res_linear_sin_long.unsqueeze(0))

        # Original Resonance NTK RoPE
        original_res_ntk = OriginalResonanceNTKScaledRotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        ).to(torch_device)
        original_res_ntk_cos_short, original_res_ntk_sin_short = original_res_ntk(x, position_ids_short.shape[-1])
        original_res_ntk_cos_long, original_res_ntk_sin_long = original_res_ntk(x, position_ids_long.shape[-1])
        torch.testing.assert_close(original_res_ntk_cos_short, original_res_ntk_cos_long[:, :, :short_input_length, :])
        torch.testing.assert_close(original_res_ntk_sin_short, original_res_ntk_sin_long[:, :, :short_input_length, :])

        # Resonance NTK RoPE
        res_ntk = ResonanceNTKScalingRotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=scaling_factor,
        ).to(torch_device)
        res_ntk_cos_short, res_ntk_sin_short = res_ntk(x, position_ids_short)
        res_ntk_cos_long, res_ntk_sin_long = res_ntk(x, position_ids_long)
        torch.testing.assert_close(res_ntk_cos_short, res_ntk_cos_long[:, :short_input_length])
        torch.testing.assert_close(res_ntk_sin_short, res_ntk_sin_long[:, :short_input_length])

        torch.testing.assert_close(original_res_ntk_cos_short, res_ntk_cos_short.unsqueeze(0))
        torch.testing.assert_close(original_res_ntk_sin_short, res_ntk_sin_short.unsqueeze(0))
        torch.testing.assert_close(original_res_ntk_cos_long, res_ntk_cos_long.unsqueeze(0))
        torch.testing.assert_close(original_res_ntk_sin_long, res_ntk_sin_long.unsqueeze(0))

        # Original Resonance YaRN RoPE
        original_res_yarn = OriginalResonanceYaRNScaledRotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=scaling_factor,
        ).to(torch_device)
        original_res_yarn_cos_short, original_res_yarn_sin_short = original_res_yarn(x, position_ids_short.shape[-1])
        original_res_yarn_cos_long, original_res_yarn_sin_long = original_res_yarn(x, position_ids_long.shape[-1])
        torch.testing.assert_close(original_res_yarn_cos_short,
                                   original_res_yarn_cos_long[:, :, :short_input_length, :])
        torch.testing.assert_close(original_res_yarn_sin_short,
                                   original_res_yarn_sin_long[:, :, :short_input_length, :])

        # Resonance YaRN RoPE
        res_yarn = ResonanceYaRNScalingRotaryEmbedding(
            head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=scaling_factor,
        ).to(torch_device)
        res_yarn_cos_short, res_yarn_sin_short = res_yarn(x, position_ids_short)
        res_yarn_cos_long, res_yarn_sin_long = res_yarn(x, position_ids_long)
        torch.testing.assert_close(res_yarn_cos_short, res_yarn_cos_long[:, :short_input_length])
        torch.testing.assert_close(res_yarn_sin_short, res_yarn_sin_long[:, :short_input_length])

        torch.testing.assert_close(original_res_yarn_cos_short, res_yarn_cos_short.unsqueeze(0))
        torch.testing.assert_close(original_res_yarn_sin_short, res_yarn_sin_short.unsqueeze(0))
        torch.testing.assert_close(original_res_yarn_cos_long, res_yarn_cos_long.unsqueeze(0))
        torch.testing.assert_close(original_res_yarn_sin_long, res_yarn_sin_long.unsqueeze(0))
