# from src.pipeline.training_pipeline import TrainingPipeline

# TrainingPipeline().run()
import sys
import os
sys.path.append(os.path.abspath("."))

print("=== Pre-flight check ===\n")

checks = {
    "DKT model"      : "artifacts/dkt_model/dkt_model.keras",
    "RL policy"      : "artifacts/rl_policy/rl_policy.pkl",
    "Question index" : "artifacts/transformed/question_index.json",
    "Train CSV"      : "artifacts/train.csv",
    "Test CSV"       : "artifacts/test.csv",
}

all_good = True
for name, path in checks.items():
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"  {status} {name:<20} : {path}")
    if not exists:
        all_good = False

print()
from src.configuration.db_connection import check_connection
check_connection()

print()
try:
    from src.utils.main_utils import load_dkt_model, load_rl_agent
    from src.constants import DKT_MODEL_PATH, RL_POLICY_PATH
    dkt = load_dkt_model(DKT_MODEL_PATH)
    print(f"  ✅ DKT model loaded: {dkt.name}")
    rl  = load_rl_agent(RL_POLICY_PATH)
    print(f"  ✅ RL agent loaded")
except Exception as e:
    print(f"  ❌ Model load failed: {e}")
    all_good = False

print()
if all_good:
    print("✅ All checks passed — safe to run uvicorn!")
else:
    print("❌ Fix issues above before running uvicorn")