import pytest


@pytest.fixture(scope="session")
def gpt2_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("gpt2")
