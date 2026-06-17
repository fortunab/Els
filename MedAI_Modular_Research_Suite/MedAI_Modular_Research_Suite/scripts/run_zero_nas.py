import argparse
from pathlib import Path
import sys
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from medai_suite.zero_nas import zero_cost_search
from medai_suite.utils import set_seed, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-architectures", type=int, default=20)
    parser.add_argument("--output", default="outputs/results/zero_nas_results.json")
    args = parser.parse_args()

    set_seed(42)
    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train[:32, ..., None].astype("float32") / 255.0

    best, records = zero_cost_search(x_train, n_architectures=args.n_architectures)
    best_model = best.pop("model")
    best_model.save("outputs/models/zero_nas_best.keras")

    save_json(args.output, {"best": best, "records": records})
    print("Best:", best)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
