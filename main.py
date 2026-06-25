from pathlib import Path

from src.data_synth import generate_customer_data
from src.analysis import run_analysis


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    data_dir = project_dir / "data"
    outputs_dir = project_dir / "outputs"
    data_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    data_path = data_dir / "customer_data.csv"
    df = generate_customer_data(str(data_path))
    run_analysis(df, outputs_dir)

    print("Projet portfolio Accenture prêt.")
    print(f"Données : {data_path}")
    print(f"Résultats : {outputs_dir}")


if __name__ == "__main__":
    main()
