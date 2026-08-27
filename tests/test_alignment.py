import torch

from llm_jb.data.alignment import extract_anchor_activations, gather_positions, mean_pool_range
from llm_jb.data.tokenize import tokenize
from llm_jb.data.types import AnchorMode, BehaviorTriple, PromptSpan


def _triple(text: str) -> BehaviorTriple:
    return BehaviorTriple(
        behavior_id=text[:8],
        category="c",
        harmful_prompt=PromptSpan.whole(text),
        benign_prompt=None,
        jailbroken_prompt=None,
        source="test",
    )


class TestMixedLengthBatchAlignment:
    """The core alignment guarantee: with right padding, the true last
    prompt token sits at a different index per row, so extraction must use
    gather on real per-row indices. `activations[:, -1, :]` would silently
    read padding for every row shorter than the longest one in the batch.
    """

    def test_gather_matches_real_last_token_not_naive_last_index(self, gpt2_tokenizer):
        short = tokenize(_triple("short prompt"), gpt2_tokenizer).harmful
        long = tokenize(
            _triple("a much longer prompt with several more words in it"), gpt2_tokenizer
        ).harmful
        assert short.input_ids.shape[0] != long.input_ids.shape[0]

        pad_id = gpt2_tokenizer.eos_token_id
        maxlen = max(short.input_ids.shape[0], long.input_ids.shape[0])
        batch_ids = torch.stack(
            [
                torch.nn.functional.pad(
                    s.input_ids, (0, maxlen - s.input_ids.shape[0]), value=pad_id
                )
                for s in (short, long)
            ]
        )
        # fake "activations": one scalar feature equal to the token id, so
        # correctness can be checked by exact value comparison
        activations = batch_ids.unsqueeze(-1).float()

        extracted = extract_anchor_activations(
            activations, [short, long], mode=AnchorMode.LAST_PROMPT_POSITION
        ).squeeze(-1)

        expected = torch.tensor(
            [short.input_ids[-1].item(), long.input_ids[-1].item()], dtype=torch.float
        )
        assert torch.equal(extracted, expected)

        naive_last_index = activations[:, -1, :].squeeze(-1)
        # the short row's naive last index is padding, not its real last token
        assert naive_last_index[0].item() == pad_id
        assert not torch.equal(naive_last_index, expected)

    def test_gather_positions_shape(self):
        activations = torch.arange(2 * 5 * 3).reshape(2, 5, 3).float()
        positions = torch.tensor([1, 4])
        out = gather_positions(activations, positions)
        assert out.shape == (2, 3)
        assert torch.equal(out[0], activations[0, 1])
        assert torch.equal(out[1], activations[1, 4])


class TestMeanPoolRange:
    def test_mean_over_known_range(self):
        activations = torch.zeros(1, 6, 1)
        activations[0, 2:5, 0] = torch.tensor([2.0, 4.0, 6.0])
        out = mean_pool_range(activations, torch.tensor([2]), torch.tensor([5]))
        assert out.shape == (1, 1)
        assert torch.isclose(out[0, 0], torch.tensor(4.0))

    def test_empty_range_does_not_divide_by_zero(self):
        activations = torch.ones(1, 4, 1)
        out = mean_pool_range(activations, torch.tensor([2]), torch.tensor([2]))
        assert torch.isfinite(out).all()


class TestExtractAnchorActivationsModes:
    def test_last_k_tokens(self, gpt2_tokenizer):
        span = tokenize(_triple("one two three four five"), gpt2_tokenizer).harmful
        activations = span.input_ids.unsqueeze(0).unsqueeze(-1).float()
        out = extract_anchor_activations(activations, [span], mode=AnchorMode.LAST_K_TOKENS, k=3)
        expected = span.input_ids[-3:].float().mean()
        assert torch.isclose(out[0, 0], expected)

    def test_mean_instruction_span_uses_only_instruction_tokens(self, gpt2_tokenizer):
        text = "before words INSTRUCTION after words"
        instr_start = text.index("INSTRUCTION")
        instr_end = instr_start + len("INSTRUCTION")
        span = PromptSpan(
            text=text,
            instruction_start=instr_start,
            instruction_end=instr_end,
            response_start=len(text),
        )
        triple = BehaviorTriple(
            behavior_id="x",
            category="c",
            harmful_prompt=span,
            benign_prompt=None,
            jailbroken_prompt=None,
            source="test",
        )
        tokenized = tokenize(triple, gpt2_tokenizer).harmful
        activations = tokenized.input_ids.unsqueeze(0).unsqueeze(-1).float()
        out = extract_anchor_activations(
            activations, [tokenized], mode=AnchorMode.MEAN_INSTRUCTION_SPAN
        )
        expected = (
            tokenized.input_ids[tokenized.instruction_token_start : tokenized.instruction_token_end]
            .float()
            .mean()
        )
        assert torch.isclose(out[0, 0], expected)
