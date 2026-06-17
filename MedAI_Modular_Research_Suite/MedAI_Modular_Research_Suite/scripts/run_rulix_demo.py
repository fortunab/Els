from pathlib import Path
import sys
import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from medai_suite.rulix import build_rulix_demo


model = build_rulix_demo()
x = torch.randn(4, 16, 128)
with torch.no_grad():
    y = model(x)
print("R-ULIx demo output shape:", tuple(y.shape))
