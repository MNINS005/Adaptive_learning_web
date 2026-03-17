import os

# ── Database ───────────────────────────────────────────────────────────
DB_USER     = os.getenv("DB_USER",     "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "yourpassword")
DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_PORT     = int(os.getenv("DB_PORT", 5432))
DB_NAME     = os.getenv("DB_NAME",     "learning_platform_project")

# ── Artifact paths ─────────────────────────────────────────────────────
ARTIFACT_DIR     = "artifacts"
DKT_MODEL_DIR    = os.path.join(ARTIFACT_DIR, "dkt_model")
RL_POLICY_DIR    = os.path.join(ARTIFACT_DIR, "rl_policy")
DKT_MODEL_PATH   = os.path.join(DKT_MODEL_DIR, "dkt_model.keras")
RL_POLICY_PATH   = os.path.join(RL_POLICY_DIR, "rl_policy.pkl")

# ── DKT model ──────────────────────────────────────────────────────────
NUM_QUESTIONS    = 50
DKT_HIDDEN_SIZE  = 128
DKT_DROPOUT      = 0.2
DKT_BATCH_SIZE   = 32
DKT_EPOCHS       = 50
DKT_LEARNING_RATE = 0.001

# ── RL agent ───────────────────────────────────────────────────────────
RL_LEARNING_RATE = 0.001
RL_GAMMA         = 0.95
RL_EPSILON       = 0.1
ZPD_DELTA        = 0.15

# ── NLP ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"
EMBEDDING_DIM    = 384

# ── AWS ────────────────────────────────────────────────────────────────
S3_BUCKET        = os.getenv("S3_BUCKET", "your-model-bucket")
S3_DKT_MODEL_KEY = "models/dkt_model.keras"
S3_RL_POLICY_KEY = "models/rl_policy.pkl"
AWS_REGION       = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")

# ── MLflow ─────────────────────────────────────────────────────────────
MLFLOW_EXPERIMENT = "adaptive_learning"

# ── Topics ─────────────────────────────────────────────────────────────
TOPICS = [
    "arrays",
    "linked_lists",
    "trees",
    "graphs",
    "dynamic_programming",
    "sorting",
    "searching",
    "recursion",
]

# ── Reward shaping ─────────────────────────────────────────────────────
REWARD_CORRECT          =  1.0
REWARD_WRONG            = -0.3
REWARD_TOO_EASY_PENALTY = -0.4
TOO_EASY_THRESHOLD      =  0.3