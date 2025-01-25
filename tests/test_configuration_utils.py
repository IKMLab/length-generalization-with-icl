# coding=utf-8
# Copyright 2019 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import shutil
import sys
import tempfile
import unittest
import unittest.mock as mock
from pathlib import Path

from huggingface_hub import HfFolder, delete_repo
from requests.exceptions import HTTPError

from transformers import AutoConfig, BertConfig, GPT2Config
from transformers.configuration_utils import PretrainedConfig
from transformers.testing_utils import TOKEN, USER, is_staging_test

config_common_kwargs = {
    "return_dict": False,
    "output_hidden_states": True,
    "output_attentions": True,
    "torchscript": True,
    "torch_dtype": "float16",
    "use_bfloat16": True,
    "tf_legacy_loss": True,
    "pruned_heads": {
        "a": 1
    },
    "tie_word_embeddings": False,
    "is_decoder": True,
    "cross_attention_hidden_size": 128,
    "add_cross_attention": True,
    "tie_encoder_decoder": True,
    "max_length": 50,
    "min_length": 3,
    "do_sample": True,
    "early_stopping": True,
    "num_beams": 3,
    "num_beam_groups": 3,
    "diversity_penalty": 0.5,
    "temperature": 2.0,
    "top_k": 10,
    "top_p": 0.7,
    "typical_p": 0.2,
    "repetition_penalty": 0.8,
    "length_penalty": 0.8,
    "no_repeat_ngram_size": 5,
    "encoder_no_repeat_ngram_size": 5,
    "bad_words_ids": [1, 2, 3],
    "num_return_sequences": 3,
    "chunk_size_feed_forward": 5,
    "output_scores": True,
    "return_dict_in_generate": True,
    "forced_bos_token_id": 2,
    "forced_eos_token_id": 3,
    "remove_invalid_values": True,
    "architectures": ["BertModel"],
    "finetuning_task": "translation",
    "id2label": {
        0: "label"
    },
    "label2id": {
        "label": "0"
    },
    "tokenizer_class": "BertTokenizerFast",
    "prefix": "prefix",
    "bos_token_id": 6,
    "pad_token_id": 7,
    "eos_token_id": 8,
    "sep_token_id": 9,
    "decoder_start_token_id": 10,
    "exponential_decay_length_penalty": (5, 1.01),
    "suppress_tokens": [0, 1],
    "begin_suppress_tokens": 2,
    "task_specific_params": {
        "translation": "some_params"
    },
    "problem_type": "regression",
}
