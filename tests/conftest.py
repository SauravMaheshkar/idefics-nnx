import pytest
from transformers import AutoModelForImageTextToText


@pytest.fixture
def get_idefics2_from_hub():
    model = AutoModelForImageTextToText.from_pretrained("HuggingFaceM4/idefics2-8b")
    return model
