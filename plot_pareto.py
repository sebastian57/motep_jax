import optuna
import os

STUDY_NAME = "jax_train4_val_var_level"
DB_PATH = "optuna_studies/jax_train4_val_var_level.db" 

study = optuna.load_study(
    study_name=STUDY_NAME,
    storage=f"sqlite:///{DB_PATH}"
)

fig = optuna.visualization.plot_pareto_front(study, target_names=["Final Loss", "Total Steps"])

PLOT_OUTPUT_DIR = "optuna_studies" 
os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
output_plot_path = os.path.join(PLOT_OUTPUT_DIR, f"{STUDY_NAME}_pareto_front.pdf")

try:
    fig.write_image(output_plot_path)
    print(f"Plot saved to {output_plot_path}")
except ImportError:
    print("Error saving plot: Please install kaleido (`pip install kaleido`)")
except Exception as e:
    print(f"Error saving plot: {e}")