import argparse
from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from medai_suite.aasag import load_mnist, build_mlp, build_cnn, build_convmixer_lite, train_model, sag_aggregate
from medai_suite.utils import save_json, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output", default="outputs/results/aasag_results.json")
    args = parser.parse_args()

    set_seed(42)
    train_ds, test_ds, (x_test, y_test) = load_mnist()

    models = [build_mlp(), build_cnn(), build_convmixer_lite()]
    names = ["mlp", "cnn", "convmixer_lite"]

    for model in models:
        train_model(model, train_ds, test_ds, epochs=args.epochs)

    image = x_test[args.sample_index]
    label = int(y_test[args.sample_index])
    logits, weights, _ = sag_aggregate(models, image, label)

    payload = {
        "label": label,
        "prediction": int(np.argmax(logits)),
        "weights": {name: float(w) for name, w in zip(names, weights)},
    }
    save_json(args.output, payload)
    print(payload)


if __name__ == "__main__":
    main()
