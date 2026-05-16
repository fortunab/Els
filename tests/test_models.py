import torch
from multih.models.factory import build_model


def test_cnn_forward():
    model = build_model("cnn", num_classes=2, image_size=64)
    y = model(torch.randn(2, 3, 64, 64))
    assert y.shape == (2, 2)
