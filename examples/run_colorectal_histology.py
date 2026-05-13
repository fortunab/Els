from hydra_aixia.colorectal_histology import ColorectalConfig, train_and_save

if __name__ == "__main__":
    model = train_and_save(ColorectalConfig(epochs=5))
    print("Model trained and saved.")
