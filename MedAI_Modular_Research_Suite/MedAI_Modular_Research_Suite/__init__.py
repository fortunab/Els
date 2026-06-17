# Model definition files

This folder contains the standalone model definitions requested for GitHub.

Included models:

```text
crc_moe_model.py          CRC Zero-Cost Pruned Mixture-of-Experts
aasag_models.py           MLP, CNN, ConvMixer-lite for aaSAG
zero_nas_model.py         Zero-cost NAS candidate generator
rulix_model.py            R-ULIx barebone PyTorch model
```

To generate untrained local model artifacts:

```bash
python scripts/export_untrained_models.py
```

Generated files:

```text
outputs/models/crc_moe_untrained.keras
outputs/models/aasag_mlp_untrained.keras
outputs/models/aasag_cnn_untrained.keras
outputs/models/aasag_convmixer_lite_untrained.keras
outputs/models/zero_nas_candidate_untrained.keras
outputs/models/rulix_untrained.pt
```
