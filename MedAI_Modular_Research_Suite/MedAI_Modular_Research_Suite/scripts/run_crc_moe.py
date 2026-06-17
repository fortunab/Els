import argparse
from pathlib import Path
import sys
import tensorflow as tf

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from medai_suite.crc_moe import build_crc_moe_experiment
from medai_suite.utils import set_seed, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--img-size", type=int, default=96)
    parser.add_argument("--output", default="outputs/results/crc_moe_results.json")
    args = parser.parse_args()

    set_seed(42)

    model, train_ds, val_ds, test_ds, scores, selected_idx = build_crc_moe_experiment(
        img_size=args.img_size,
        num_experts_pool=8,
        topk_experts=4,
        topk_route=2,
        score_batches=1,
        seed=42,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)
    test_loss, test_acc = model.evaluate(test_ds)

    model.save("outputs/models/crc_moe.keras")

    payload = {
        "zero_cost_scores": scores,
        "selected_experts": selected_idx,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "history": {k: [float(vv) for vv in v] for k, v in history.history.items()},
    }
    save_json(args.output, payload)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
