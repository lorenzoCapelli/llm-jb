import os
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from llm_jb.models.backend import load_model, load_model_nnsight, resolve_device
from llm_jb.models.config import ModelConfig

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs" / "model"


class TestModelConfig:
    def test_loads_gpt2_small_yaml(self):
        config = ModelConfig.load(CONFIGS_DIR / "gpt2-small.yaml")
        assert config.name == "gpt2-small"
        assert config.hf_repo_id == "gpt2"
        assert config.backend == "transformer_lens"

    def test_loads_llama_yaml_without_downloading(self):
        config = ModelConfig.load(CONFIGS_DIR / "llama-3.2-1b-instruct.yaml")
        assert config.hf_repo_id == "meta-llama/Llama-3.2-1B-Instruct"
        assert config.dtype == "bfloat16"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ModelConfig.load(tmp_path / "does-not-exist.yaml")

    def test_env_var_overrides_yaml(self, monkeypatch):
        monkeypatch.setenv("LLM_JB_MODEL_DEVICE", "cpu")
        config = ModelConfig.load(CONFIGS_DIR / "gpt2-small.yaml")
        assert config.device == "cpu"

    def test_rejects_unknown_field(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text("name: x\nhf_repo_id: gpt2\nnot_a_real_field: 1\n")
        with pytest.raises(ValidationError):
            ModelConfig.load(bad)


class TestResolveDevice:
    def test_explicit_device_passthrough(self):
        assert resolve_device("cpu") == "cpu"
        assert resolve_device("cuda:2") == "cuda:2"

    def test_auto_resolves_by_availability(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert resolve_device("auto") == "cpu"
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert resolve_device("auto") == "cuda"


class TestBackendDispatch:
    def test_nnsight_stub_raises_not_implemented(self):
        config = ModelConfig(name="x", hf_repo_id="y", backend="nnsight")
        with pytest.raises(NotImplementedError):
            load_model_nnsight(config)

    def test_unknown_backend_raises(self):
        config = ModelConfig.model_construct(
            name="x", hf_repo_id="y", backend="not-a-backend", device="cpu", dtype="float32"
        )
        with pytest.raises(ValueError):
            load_model(config)


class TestLoadGpt2Small:
    """Real (small, CPU) end-to-end load — validates the whole config ->
    backend -> HookedTransformer path, not just the dispatch logic."""

    def test_loads_and_runs_forward_pass(self):
        config = ModelConfig.load(CONFIGS_DIR / "gpt2-small.yaml")
        os.environ.pop("LLM_JB_MODEL_DEVICE", None)
        model = load_model(config)
        assert model.cfg.n_layers == 12
        tokens = model.to_tokens("Hello, world!")
        logits = model(tokens)
        assert logits.shape[0] == 1
        assert logits.shape[1] == tokens.shape[1]
        assert logits.shape[2] == model.cfg.d_vocab
