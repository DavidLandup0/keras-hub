import os

import pytest

from keras_hub.src.models.f_net.f_net_text_classifier_preprocessor import (
    FNetTextClassifierPreprocessor,
)
from keras_hub.src.models.f_net.f_net_tokenizer import FNetTokenizer
from keras_hub.src.tests.test_case import TestCase


class FNetTextClassifierPreprocessorTest(TestCase):
    def setUp(self):
        self.tokenizer = FNetTokenizer(
            # Generated using create_f_net_test_proto.py
            proto=os.path.join(self.get_test_data_dir(), "f_net_test_vocab.spm")
        )
        self.init_kwargs = {
            "tokenizer": self.tokenizer,
            "sequence_length": 8,
        }
        self.input_data = (
            ["the quick brown fox"],
            [1],  # Pass through labels.
            [1.0],  # Pass through sample_weights.
        )

    def test_preprocessor_basics(self):
        self.run_preprocessor_test(
            cls=FNetTextClassifierPreprocessor,
            init_kwargs=self.init_kwargs,
            input_data=self.input_data,
            expected_output=(
                {
                    "token_ids": [[2, 5, 10, 6, 8, 3, 0, 0]],
                    "segment_ids": [[0, 0, 0, 0, 0, 0, 0, 0]],
                },
                [1],  # Pass through labels.
                [1.0],  # Pass through sample_weights.
            ),
        )

    def test_errors_for_2d_list_input(self):
        preprocessor = FNetTextClassifierPreprocessor(**self.init_kwargs)
        ambiguous_input = [["one", "two"], ["three", "four"]]
        with self.assertRaises(ValueError):
            preprocessor(ambiguous_input)

    @pytest.mark.extra_large
    def test_all_presets(self):
        for preset in FNetTextClassifierPreprocessor.presets:
            self.run_preset_test(
                cls=FNetTextClassifierPreprocessor,
                preset=preset,
                input_data=self.input_data,
            )