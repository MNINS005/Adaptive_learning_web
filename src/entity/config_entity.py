from dataclasses import dataclass
from src.constants import (
    ARTIFACT_DIR, DKT_HIDDEN_SIZE, DKT_DROPOUT, DKT_BATCH_SIZE,
    DKT_EPOCHS, DKT_LEARNING_RATE, NUM_QUESTIONS,
    DKT_MODEL_PATH, RL_POLICY_PATH,
    EMBEDDING_MODEL, EMBEDDING_DIM,
    MLFLOW_EXPERIMENT
)

@dataclass
class DataIngestionConfig:
    artifact_dir:          str   = ARTIFACT_DIR
    min_attempts_to_train: int   = 500
    test_size:             float = 0.2
    random_state:          int   = 42
@dataclass
class DataValidationConfig:
    required_columns: tuple = ("user_id", "question_id", "is_correct")
    min_unique_users: int   = 10

@dataclass
class DKTTrainerConfig:
    num_questions:  int   = NUM_QUESTIONS
    hidden_size:    int   = DKT_HIDDEN_SIZE
    dropout:        float = DKT_DROPOUT
    batch_size:     int   = DKT_BATCH_SIZE
    epochs:         int   = DKT_EPOCHS
    learning_rate:  float = DKT_LEARNING_RATE
    model_path:     str   = DKT_MODEL_PATH

@dataclass
class RLTrainerConfig:
    policy_path:    str   = RL_POLICY_PATH
    learning_rate:  float = 0.001
    episodes:       int   = 1000

@dataclass
class ModelEvaluationConfig:
    mlflow_experiment: str   = MLFLOW_EXPERIMENT
    auc_threshold:     float = 0.75   # min AUC to promote model

@dataclass
class ModelPusherConfig:
    s3_dkt_key: str = "models/dkt_model.keras"
    s3_rl_key:  str = "models/rl_policy.pkl"