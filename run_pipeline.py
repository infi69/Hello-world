from pathlib import Path
from main import run_pipeline

crops_dir = Path(r"C:\Users\Anant.Jain\OneDrive - PACCAR Inc\Documents\AI_Initatives\Bar_code_detection\20260910-121804-e7c87ffd\media\9d2f042a-3c6c-4d3c-9708-64c72f1fe53a\artifacts")
results_dir = Path(r"C:\Users\Anant.Jain\OneDrive - PACCAR Inc\Documents\AI_Initatives\Bar_code_detection\pallete_results")

results_dir.mkdir(parents=True, exist_ok=True)

run_pipeline(
    crops_dir=crops_dir,
    results_dir=results_dir,
    workers=1,
    batch_size=1,
    aggressive_max_passes=12,
)

print("CSV files in results dir:")
print(list(results_dir.glob("*.csv")))