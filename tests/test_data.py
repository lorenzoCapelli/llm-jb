import pytest

from llm_jb.data.loaders.advbench import _records_to_triples as advbench_to_triples
from llm_jb.data.loaders.harmbench import _records_to_triples as harmbench_to_triples
from llm_jb.data.loaders.jbb import _make_jailbroken_span
from llm_jb.data.loaders.jbb import _records_to_triples as jbb_to_triples
from llm_jb.data.tokenize import tokenize
from llm_jb.data.types import PromptSpan


class TestPromptSpan:
    def test_whole(self):
        span = PromptSpan.whole("hello world")
        assert span.instruction_start == 0
        assert span.instruction_end == len("hello world")
        assert span.response_start == len("hello world")

    def test_rejects_out_of_order_span(self):
        with pytest.raises(ValueError):
            PromptSpan(text="abc", instruction_start=2, instruction_end=1, response_start=3)

    def test_rejects_instruction_end_past_text(self):
        with pytest.raises(ValueError):
            PromptSpan(text="abc", instruction_start=0, instruction_end=10, response_start=10)

    def test_rejects_response_start_before_instruction_end(self):
        with pytest.raises(ValueError):
            PromptSpan(text="abcdef", instruction_start=0, instruction_end=4, response_start=2)


class TestJbbLoader:
    def test_make_jailbroken_span_exact_match(self):
        span, matched = _make_jailbroken_span("prefix GOAL suffix", "GOAL")
        assert matched
        assert span.text[span.instruction_start : span.instruction_end] == "GOAL"

    def test_make_jailbroken_span_no_match_falls_back_to_whole(self):
        span, matched = _make_jailbroken_span("a paraphrased version of it", "GOAL")
        assert not matched
        assert span.instruction_start == 0
        assert span.instruction_end == len(span.text)

    def test_records_to_triples_pairs_by_index(self):
        harmful = {0: {"Category": "cat", "Goal": "do harm", "Behavior": "Harm", "Target": "Sure"}}
        benign = {0: {"Goal": "do good"}}
        artifacts = {0: {"prompt": "wrapper do harm wrapper", "jailbroken": True}}
        triples = jbb_to_triples(harmful, benign, artifacts, "PAIR", "some-model")
        assert len(triples) == 1
        t = triples[0]
        assert t.behavior_id == "jbb_0"
        assert t.harmful_prompt.text == "do harm"
        assert t.benign_prompt.text == "do good"
        assert t.jailbroken_prompt is not None
        assert t.metadata["instruction_span_matched"] is True

    def test_records_to_triples_missing_artifact_is_none(self):
        harmful = {0: {"Category": "cat", "Goal": "do harm", "Behavior": "Harm", "Target": "Sure"}}
        triples = jbb_to_triples(harmful, {}, {}, "PAIR", "some-model")
        assert triples[0].benign_prompt is None
        assert triples[0].jailbroken_prompt is None
        assert triples[0].metadata["instruction_span_matched"] is None


class TestAdvbenchLoader:
    def test_records_to_triples(self):
        records = [{"prompt": "do the bad thing", "target": "Sure, here is"}]
        triples = advbench_to_triples(records)
        assert len(triples) == 1
        assert triples[0].behavior_id == "advbench_0"
        assert triples[0].source == "advbench"
        assert triples[0].harmful_prompt.text == "do the bad thing"
        assert triples[0].metadata["target"] == "Sure, here is"


class TestHarmbenchLoader:
    def test_records_to_triples(self):
        records = [{"prompt": "do the bad thing", "category": "chemical"}]
        triples = harmbench_to_triples(records, config="standard")
        assert len(triples) == 1
        assert triples[0].behavior_id == "harmbench_standard_0"
        assert triples[0].category == "chemical"

    def test_records_to_triples_missing_category_defaults(self):
        records = [{"prompt": "do the bad thing"}]
        triples = harmbench_to_triples(records, config="standard")
        assert triples[0].category == "uncategorized"


class TestTokenize:
    def test_instruction_span_maps_into_wrapper(self, gpt2_tokenizer):
        text = "prefix words here INSTRUCTION more words after"
        instr_start = text.index("INSTRUCTION")
        instr_end = instr_start + len("INSTRUCTION")
        span = PromptSpan(
            text=text,
            instruction_start=instr_start,
            instruction_end=instr_end,
            response_start=len(text),
        )
        from llm_jb.data.types import BehaviorTriple

        triple = BehaviorTriple(
            behavior_id="x",
            category="c",
            harmful_prompt=span,
            benign_prompt=None,
            jailbroken_prompt=None,
            source="test",
        )
        tokenized = tokenize(triple, gpt2_tokenizer)
        harmful = tokenized.harmful

        decoded_instruction = gpt2_tokenizer.decode(
            harmful.input_ids[harmful.instruction_token_start : harmful.instruction_token_end]
        )
        assert "INSTRUCTION" in decoded_instruction
        # tokens outside the instruction span shouldn't contain the marker word
        before = gpt2_tokenizer.decode(harmful.input_ids[: harmful.instruction_token_start])
        after = gpt2_tokenizer.decode(harmful.input_ids[harmful.instruction_token_end :])
        assert "INSTRUCTION" not in before
        assert "INSTRUCTION" not in after

    def test_last_prompt_position_is_last_token(self, gpt2_tokenizer):
        from llm_jb.data.types import BehaviorTriple

        span = PromptSpan.whole("a short prompt")
        triple = BehaviorTriple(
            behavior_id="x",
            category="c",
            harmful_prompt=span,
            benign_prompt=None,
            jailbroken_prompt=None,
            source="test",
        )
        tokenized = tokenize(triple, gpt2_tokenizer)
        assert tokenized.harmful.last_prompt_position == tokenized.harmful.input_ids.shape[0] - 1

    def test_chat_template_branch_offsets_full_templated_text(self, gpt2_tokenizer):
        from llm_jb.data.types import BehaviorTriple

        class TemplatedTokenizer:
            """Fakes a chat template around a real tokenizer's __call__, so
            we can test the templated branch without downloading an
            instruct model just for its chat_template."""

            def __init__(self, base):
                self._base = base
                self.chat_template = "fake-template"

            def apply_chat_template(self, conversation, tokenize, add_generation_prompt):
                content = conversation[0]["content"]
                return f"<|user|>\n{content}\n<|assistant|>\n"

            def __call__(self, text, return_offsets_mapping, return_tensors):
                return self._base(
                    text,
                    return_offsets_mapping=return_offsets_mapping,
                    return_tensors=return_tensors,
                )

            def decode(self, ids):
                return self._base.decode(ids)

        templated = TemplatedTokenizer(gpt2_tokenizer)
        span = PromptSpan.whole("what is the capital of France")
        triple = BehaviorTriple(
            behavior_id="x",
            category="c",
            harmful_prompt=span,
            benign_prompt=None,
            jailbroken_prompt=None,
            source="test",
        )
        tokenized = tokenize(triple, templated)
        harmful = tokenized.harmful

        decoded_instruction = templated.decode(
            harmful.input_ids[harmful.instruction_token_start : harmful.instruction_token_end]
        )
        assert "France" in decoded_instruction
        # the assistant-turn opening comes after the instruction and before
        # the last prompt position, so last_prompt_position must sit past
        # the instruction span
        assert tokenized.harmful.last_prompt_position >= harmful.instruction_token_end
