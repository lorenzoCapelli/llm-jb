import pytest
import torch
from transformer_lens import HookedTransformer

from llm_jb.data.tokenize import tokenize
from llm_jb.data.types import AnchorMode, BehaviorTriple, PromptSpan
from llm_jb.hooks.capture import capture_residual_stream
from llm_jb.hooks.storage import (
    load_activations_safetensors,
    load_activations_zarr,
    place_activation,
    save_activations_safetensors,
    save_activations_zarr,
)

# Configurable VRAM budget for a small capture batch — tune per hardware/model.
MAX_CAPTURE_VRAM_DELTA_BYTES = 200 * 1024 * 1024  # 200 MB


@pytest.fixture(scope="module")
def gpt2_model():
    return HookedTransformer.from_pretrained("gpt2", device="cpu")


def _make_batch(model: HookedTransformer, texts: list[str]):
    spans = []
    for text in texts:
        triple = BehaviorTriple(
            behavior_id=text[:8],
            category="c",
            harmful_prompt=PromptSpan.whole(text),
            benign_prompt=None,
            jailbroken_prompt=None,
            source="test",
        )
        spans.append(tokenize(triple, model.tokenizer).harmful)

    pad_id = model.tokenizer.eos_token_id
    maxlen = max(s.input_ids.shape[0] for s in spans)
    batch = torch.stack(
        [
            torch.nn.functional.pad(s.input_ids, (0, maxlen - s.input_ids.shape[0]), value=pad_id)
            for s in spans
        ]
    )
    return batch, spans


class TestCaptureResidualStream:
    def test_shape_and_dtype(self, gpt2_model):
        batch, spans = _make_batch(gpt2_model, ["short prompt", "a somewhat longer prompt here"])
        result = capture_residual_stream(gpt2_model, batch, spans, layers=[0, 5, 11])
        assert result.layers == [0, 5, 11]
        assert set(result.activations.keys()) == {0, 5, 11}
        for layer in (0, 5, 11):
            act = result.activations[layer]
            assert act.shape == (2, gpt2_model.cfg.d_model)
            assert act.dtype == torch.float32
        assert result.d_model == gpt2_model.cfg.d_model
        assert result.batch_size == 2

    def test_only_requested_layers_captured(self, gpt2_model):
        batch, spans = _make_batch(gpt2_model, ["one prompt"])
        result = capture_residual_stream(gpt2_model, batch, spans, layers=[3])
        assert list(result.activations.keys()) == [3]

    def test_deterministic(self, gpt2_model):
        batch, spans = _make_batch(gpt2_model, ["deterministic check prompt"])
        r1 = capture_residual_stream(gpt2_model, batch, spans, layers=[0, 6])
        r2 = capture_residual_stream(gpt2_model, batch, spans, layers=[0, 6])
        for layer in (0, 6):
            assert torch.equal(r1.activations[layer], r2.activations[layer])

    def test_mismatched_batch_size_raises(self, gpt2_model):
        batch, spans = _make_batch(gpt2_model, ["a", "b"])
        with pytest.raises(ValueError):
            capture_residual_stream(gpt2_model, batch, spans[:1], layers=[0])

    def test_empty_layers_raises(self, gpt2_model):
        batch, spans = _make_batch(gpt2_model, ["a"])
        with pytest.raises(ValueError):
            capture_residual_stream(gpt2_model, batch, spans, layers=[])

    def test_cpu_placement(self, gpt2_model):
        batch, spans = _make_batch(gpt2_model, ["a"])
        result = capture_residual_stream(gpt2_model, batch, spans, layers=[0], placement="cpu")
        assert result.activations[0].device.type == "cpu"

    def test_mean_instruction_span_anchor(self, gpt2_model):
        batch, spans = _make_batch(gpt2_model, ["one two three four five"])
        result = capture_residual_stream(
            gpt2_model, batch, spans, layers=[0], anchor_mode=AnchorMode.MEAN_INSTRUCTION_SPAN
        )
        assert result.activations[0].shape == (1, gpt2_model.cfg.d_model)


class TestStorageRoundTrip:
    def test_safetensors_round_trip(self, tmp_path):
        activations = {"layer_0": torch.randn(2, 8), "layer_1": torch.randn(2, 8)}
        path = tmp_path / "acts.safetensors"
        save_activations_safetensors(activations, path)
        loaded = load_activations_safetensors(path)
        for key in activations:
            assert torch.equal(loaded[key], activations[key])
            assert loaded[key].dtype == activations[key].dtype

    def test_zarr_round_trip(self, tmp_path):
        activations = {"layer_0": torch.randn(2, 8), "layer_1": torch.randn(2, 8)}
        path = tmp_path / "acts.zarr"
        save_activations_zarr(activations, path)
        loaded = load_activations_zarr(path)
        for key in activations:
            assert torch.allclose(loaded[key], activations[key])

    def test_zarr_upcasts_bfloat16(self, tmp_path):
        activations = {"layer_0": torch.randn(2, 8).to(torch.bfloat16)}
        path = tmp_path / "acts_bf16.zarr"
        save_activations_zarr(activations, path)
        loaded = load_activations_zarr(path)
        assert loaded["layer_0"].dtype == torch.float32

    def test_place_activation_invalid_raises(self):
        with pytest.raises(ValueError):
            place_activation(torch.zeros(2), "tpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU visible")
class TestVramPeak:
    def test_capture_stays_under_budget(self):
        model = HookedTransformer.from_pretrained("gpt2", device="cuda")
        batch, spans = _make_batch(
            model,
            [
                "prompt number one is here",
                "a slightly different second prompt for the batch",
                "third one, medium length, for variety in the batch",
                "fourth and final short one",
            ],
        )
        batch = batch.to("cuda")

        device = torch.device("cuda")
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)

        capture_residual_stream(model, batch, spans, layers=list(range(model.cfg.n_layers)))

        peak = torch.cuda.max_memory_allocated(device)
        delta = peak - baseline
        assert delta < MAX_CAPTURE_VRAM_DELTA_BYTES, (
            f"capture used {delta / 1024**2:.1f} MB, budget is "
            f"{MAX_CAPTURE_VRAM_DELTA_BYTES / 1024**2:.0f} MB"
        )
