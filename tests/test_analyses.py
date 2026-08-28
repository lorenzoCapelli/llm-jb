import pytest
from transformer_lens import HookedTransformer

from llm_jb.analyses import REGISTRY, Analysis, build_batch
from llm_jb.analyses.residual_capture import ResidualCaptureAnalysis, ResidualCaptureConfig
from llm_jb.data.types import AnchorMode, BehaviorTriple, PromptSpan


def _triple(behavior_id: str, harmful: str, benign: str | None = None) -> BehaviorTriple:
    return BehaviorTriple(
        behavior_id=behavior_id,
        category="c",
        harmful_prompt=PromptSpan.whole(harmful),
        benign_prompt=PromptSpan.whole(benign) if benign else None,
        jailbroken_prompt=None,
        source="test",
    )


@pytest.fixture(scope="module")
def gpt2_model():
    return HookedTransformer.from_pretrained("gpt2", device="cpu")


class TestAnalysisInterface:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            Analysis()

    def test_registry_covers_all_analyses(self):
        assert set(REGISTRY.keys()) == {
            "residual_capture",
            "logit_lens",
            "activation_patching",
            "linear_probe",
            "sae",
        }
        for cls in REGISTRY.values():
            assert issubclass(cls, Analysis)


class TestBuildBatch:
    def test_single_variant(self, gpt2_model):
        triples = [_triple("a", "short prompt"), _triple("b", "a somewhat longer prompt here")]
        batch = build_batch(triples, gpt2_model.tokenizer, variants=("harmful",))
        assert batch.tokens.shape[0] == 2
        assert batch.behavior_ids == ["a", "b"]
        assert batch.variants == ["harmful", "harmful"]
        assert len(batch.spans) == 2

    def test_multiple_variants_flatten_into_rows(self, gpt2_model):
        triples = [_triple("a", "harmful text", benign="benign text")]
        batch = build_batch(triples, gpt2_model.tokenizer, variants=("harmful", "benign"))
        assert batch.tokens.shape[0] == 2
        assert batch.behavior_ids == ["a", "a"]
        assert batch.variants == ["harmful", "benign"]

    def test_skips_missing_variant(self, gpt2_model):
        triples = [_triple("a", "harmful only")]
        batch = build_batch(triples, gpt2_model.tokenizer, variants=("harmful", "benign"))
        assert batch.tokens.shape[0] == 1
        assert batch.variants == ["harmful"]

    def test_unknown_variant_raises(self, gpt2_model):
        with pytest.raises(ValueError):
            build_batch([_triple("a", "x")], gpt2_model.tokenizer, variants=("not-a-variant",))

    def test_no_rows_raises(self, gpt2_model):
        triples = [_triple("a", "harmful only")]
        with pytest.raises(ValueError):
            build_batch(triples, gpt2_model.tokenizer, variants=("benign",))

    def test_padding_shape(self, gpt2_model):
        triples = [_triple("a", "x"), _triple("b", "a much longer prompt with many more tokens")]
        batch = build_batch(triples, gpt2_model.tokenizer)
        maxlen = max(s.input_ids.shape[0] for s in batch.spans)
        assert batch.tokens.shape == (2, maxlen)


class TestResidualCaptureAnalysis:
    def test_run_returns_expected_result(self, gpt2_model):
        triples = [_triple("a", "harmful text", benign="benign text")]
        batch = build_batch(triples, gpt2_model.tokenizer, variants=("harmful", "benign"))

        analysis = ResidualCaptureAnalysis(ResidualCaptureConfig(layers=[0, 5]))
        result = analysis.run(gpt2_model, batch)

        assert result.analysis_name == "residual_capture"
        assert result.behavior_ids == ["a", "a"]
        assert result.variants == ["harmful", "benign"]
        assert set(result.data.keys()) == {"layer_0", "layer_5"}
        for tensor in result.data.values():
            assert tensor.shape == (2, gpt2_model.cfg.d_model)
        assert result.metadata["layers"] == [0, 5]
        assert result.metadata["anchor_mode"] == AnchorMode.LAST_PROMPT_POSITION.value

    def test_default_layers_is_all_layers(self, gpt2_model):
        triples = [_triple("a", "x")]
        batch = build_batch(triples, gpt2_model.tokenizer)
        result = ResidualCaptureAnalysis().run(gpt2_model, batch)
        assert len(result.data) == gpt2_model.cfg.n_layers


class TestStubsRaiseNotImplemented:
    @pytest.mark.parametrize("name", ["logit_lens", "activation_patching", "linear_probe", "sae"])
    def test_stub_raises(self, name, gpt2_model):
        triples = [_triple("a", "x")]
        batch = build_batch(triples, gpt2_model.tokenizer)
        analysis = REGISTRY[name]()
        with pytest.raises(NotImplementedError):
            analysis.run(gpt2_model, batch)
