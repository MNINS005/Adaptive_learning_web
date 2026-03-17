import os
import sys

from src.logger import logger
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.dkt_trainer import DKTTrainer
from src.components.rl_trainer import RLTrainer
from src.components.model_evaluation import ModelEvaluation

from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    DKTTrainerConfig,
    RLTrainerConfig,
    ModelEvaluationConfig
)
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    DKTTrainerArtifact,
    RLTrainerArtifact,
    ModelEvaluationArtifact
)


class TrainingPipeline:

    def __init__(self):
        self.ingestion_config      = DataIngestionConfig()
        self.validation_config     = DataValidationConfig()
        self.transformation_config = DataTransformationConfig()
        self.dkt_config            = DKTTrainerConfig()
        self.rl_config             = RLTrainerConfig()
        self.eval_config           = ModelEvaluationConfig()

    # ── Stage 1 ────────────────────────────────────────────────────────
    def run_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("=" * 50)
            logger.info("Stage 1: Data Ingestion started")
            ingestion = DataIngestion(self.ingestion_config)
            artifact  = ingestion.initiate_data_ingestion()
            logger.info(f"Stage 1 done: {artifact.total_records} records")
            return artifact
        except Exception as e:
            raise CustomException(e, sys)

    # ── Stage 2 ────────────────────────────────────────────────────────
    def run_data_validation(
        self, ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        try:
            logger.info("=" * 50)
            logger.info("Stage 2: Data Validation started")
            validation = DataValidation(self.validation_config)
            artifact   = validation.initiate_data_validation(ingestion_artifact)
            logger.info(f"Stage 2 done: {artifact.message}")

            if not artifact.validation_status:
                raise CustomException(
                    f"Validation failed: {artifact.message}", sys
                )
            return artifact
        except Exception as e:
            raise CustomException(e, sys)

    # ── Stage 3 ────────────────────────────────────────────────────────
    def run_data_transformation(
        self, ingestion_artifact: DataIngestionArtifact
    ) -> DataTransformationArtifact:
        try:
            logger.info("=" * 50)
            logger.info("Stage 3: Data Transformation started")
            transformation = DataTransformation(self.transformation_config)
            artifact       = transformation.initiate_data_transformation(
                ingestion_artifact
            )
            logger.info(f"Stage 3 done: {artifact.num_questions} questions")
            return artifact
        except Exception as e:
            raise CustomException(e, sys)

    # ── Stage 4 ────────────────────────────────────────────────────────
    def run_dkt_training(
        self, trans_artifact: DataTransformationArtifact
    ) -> DKTTrainerArtifact:
        try:
            logger.info("=" * 50)
            logger.info("Stage 4: DKT Training started")
            trainer  = DKTTrainer(self.dkt_config)
            artifact = trainer.initiate_dkt_training(trans_artifact)
            logger.info(f"Stage 4 done: auc={artifact.val_auc}")
            return artifact
        except Exception as e:
            raise CustomException(e, sys)

    # ── Stage 5 ────────────────────────────────────────────────────────
    def run_rl_training(
        self,
        trans_artifact : DataTransformationArtifact,
        dkt_artifact   : DKTTrainerArtifact
    ) -> RLTrainerArtifact:
        try:
            logger.info("=" * 50)
            logger.info("Stage 5: RL Training started")
            trainer  = RLTrainer(self.rl_config)
            artifact = trainer.initiate_rl_training(
                trans_artifact, dkt_artifact
            )
            logger.info(f"Stage 5 done: avg_reward={artifact.avg_reward}")
            return artifact
        except Exception as e:
            raise CustomException(e, sys)

    # ── Stage 6 ────────────────────────────────────────────────────────
    def run_model_evaluation(
        self,
        trans_artifact : DataTransformationArtifact,
        dkt_artifact   : DKTTrainerArtifact,
        rl_artifact    : RLTrainerArtifact
    ) -> ModelEvaluationArtifact:
        try:
            logger.info("=" * 50)
            logger.info("Stage 6: Model Evaluation started")
            evaluator = ModelEvaluation(self.eval_config)
            artifact  = evaluator.initiate_model_evaluation(
                trans_artifact, dkt_artifact, rl_artifact
            )
            logger.info(f"Stage 6 done: accepted={artifact.is_model_accepted}")
            return artifact
        except Exception as e:
            raise CustomException(e, sys)

    # ── run full pipeline ──────────────────────────────────────────────
    def run(self):
        try:
            logger.info("*" * 50)
            logger.info("TRAINING PIPELINE STARTED")
            logger.info("*" * 50)

            # stage 1
            ingestion_artifact    = self.run_data_ingestion()

            # stage 2
            validation_artifact   = self.run_data_validation(
                ingestion_artifact
            )

            # stage 3
            trans_artifact        = self.run_data_transformation(
                ingestion_artifact
            )

            # stage 4
            dkt_artifact          = self.run_dkt_training(trans_artifact)

            # stage 5
            rl_artifact           = self.run_rl_training(
                trans_artifact, dkt_artifact
            )

            # stage 6
            eval_artifact         = self.run_model_evaluation(
                trans_artifact, dkt_artifact, rl_artifact
            )

            logger.info("*" * 50)
            logger.info("TRAINING PIPELINE COMPLETED")
            logger.info("*" * 50)

            # print final summary
            print("\n" + "=" * 50)
            print("TRAINING PIPELINE SUMMARY")
            print("=" * 50)
            print(f"Total records     : {ingestion_artifact.total_records}")
            print(f"Validation        : {validation_artifact.validation_status}")
            print(f"Num questions     : {trans_artifact.num_questions}")
            print(f"DKT val AUC       : {dkt_artifact.val_auc}")
            print(f"DKT val loss      : {dkt_artifact.val_loss}")
            print(f"RL avg reward     : {rl_artifact.avg_reward}")
            print(f"Model accepted    : {eval_artifact.is_model_accepted}")
            print(f"Message           : {eval_artifact.message}")
            print("=" * 50)

            return eval_artifact

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise CustomException(e, sys)