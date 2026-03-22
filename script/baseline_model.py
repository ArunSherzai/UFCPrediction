from model_utils import train_all_models


def train_baseline_model():
    print("Trainiere Baseline- und Vergleichsmodelle...")
    artifacts = train_all_models()
    metrics = artifacts["winner_baseline"]["metrics"]

    print("\nBaseline Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    train_baseline_model()
