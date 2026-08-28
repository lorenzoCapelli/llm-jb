import pytest

from llm_jb.metrics import (
    JUDGE_REGISTRY,
    Judge,
    ModelJudge,
    SubstringRefusalJudge,
    Verdict,
)


class TestSubstringRefusalJudge:
    def test_flags_known_refusal(self):
        judge = SubstringRefusalJudge()
        result = judge.judge("I'm sorry, but I cannot help with that request.")
        assert result.verdict == Verdict.REFUSAL
        assert result.backend == "substring"
        assert result.evidence is not None

    def test_flags_compliant_response(self):
        judge = SubstringRefusalJudge()
        result = judge.judge("Sure, here is a step-by-step guide to baking bread.")
        assert result.verdict == Verdict.COMPLIANCE
        assert result.evidence is None

    def test_case_insensitive(self):
        judge = SubstringRefusalJudge()
        result = judge.judge("i CANNOT provide that information.")
        assert result.verdict == Verdict.REFUSAL

    def test_custom_patterns_override_default(self):
        judge = SubstringRefusalJudge(patterns=["absolutely not"])
        assert judge.judge("I'm sorry, I cannot help.").verdict == Verdict.COMPLIANCE
        assert judge.judge("Absolutely not, that's dangerous.").verdict == Verdict.REFUSAL

    def test_judge_batch(self):
        judge = SubstringRefusalJudge()
        results = judge.judge_batch(
            ["I cannot assist with that.", "Sure, here you go: step one is..."]
        )
        assert [r.verdict for r in results] == [Verdict.REFUSAL, Verdict.COMPLIANCE]


class TestModelJudgeStub:
    def test_raises_not_implemented(self):
        judge = ModelJudge()
        with pytest.raises(NotImplementedError):
            judge.judge("any response")


class TestJudgeRegistry:
    def test_registry_covers_both_backends(self):
        assert set(JUDGE_REGISTRY.keys()) == {"substring", "model"}
        for cls in JUDGE_REGISTRY.values():
            assert issubclass(cls, Judge)

    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            Judge()
