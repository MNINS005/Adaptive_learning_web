from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    train_data_path: str
    test_data_path:  str
    total_records:   int

@dataclass
class DataValidationArtifact:
    validation_status: bool
    message:           str

@dataclass
class DataTransformationArtifact:
    transformed_train_path: str
    transformed_test_path:  str
    encoder_path:           str
    num_questions:          int

@dataclass
class DKTTrainerArtifact:
    model_path: str
    val_auc:    float
    val_loss:   float


@dataclass
class RLTrainerArtifact:
    policy_path:   str
    avg_reward:    float
    episodes:    int

@dataclass
class ModelEvaluationArtifact:
    is_model_accepted: bool
    trained_auc:       float
    best_auc:          float
    message:           str


@dataclass
class ModelPusherArtifact:
    s3_dkt_path: str
    s3_rl_path:  str