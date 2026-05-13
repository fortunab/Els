from hydra_aixia.mnist_resnet_cv import run_cross_validation

if __name__ == "__main__":
    score = run_cross_validation(folds=10, epochs=5)
    print(f"Average accuracy across folds: {score:.4f}")
