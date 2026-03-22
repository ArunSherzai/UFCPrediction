from model_utils import train_all_models


def train_and_save_models():
    print("Starte Training der UFC-Modelle mit Zeit-Split-Evaluation...")
    artifacts = train_all_models()

    summary = artifacts["summary"]
    print("\nWinner Baseline Metrics:")
    for key, value in summary["winner_baseline"].items():
        print(f"  {key}: {value}")

    print("\nWinner Advanced Metrics:")
    for key, value in summary["winner_advanced"].items():
        print(f"  {key}: {value}")

    if summary["round_model"]:
        print("\nRound Model Metrics:")
        for key, value in summary["round_model"].items():
            print(f"  {key}: {value}")
    else:
        print("\nRound Model wurde wegen zu weniger Daten nicht trainiert.")

    if summary["method_model"]:
        print("\nMethod Model Metrics:")
        for key, value in summary["method_model"].items():
            print(f"  {key}: {value}")
    else:
        print("\nMethod Model wurde wegen zu weniger Daten nicht trainiert.")

    print(f"\nEmpfohlenes Sieger-Modell: {summary['recommended_winner_model']}")


if __name__ == "__main__":
    train_and_save_models()
