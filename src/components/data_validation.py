import sys
import pandas as pd

from src.logger import logger
from src.exception import CustomException
from src.entity.config_entity import DataValidationConfig
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact
)


class DataValidation:

    def __init__(self, config: DataValidationConfig = DataValidationConfig()):
        self.config = config

    def validate_columns(self, df: pd.DataFrame) -> bool:
        try:
            missing = [
                col for col in self.config.required_columns
                if col not in df.columns
            ]
            if missing:
                logger.error(f"Missing columns: {missing}")
                return False
            logger.info("Column validation passed")
            return True

        except Exception as e:
            raise CustomException(e, sys)

    def validate_min_users(self, df: pd.DataFrame) -> bool:
        try:
            unique_users = df["user_id"].nunique()
            if unique_users < self.config.min_unique_users:
                logger.error(
                    f"Not enough users: {unique_users} < {self.config.min_unique_users}"
                )
                return False
            logger.info(f"User validation passed: {unique_users} unique users")
            return True

        except Exception as e:
            raise CustomException(e, sys)

    def validate_min_attempts(self, df: pd.DataFrame) -> bool:
        try:
            if len(df) < self.config.min_attempts:
                logger.error(
                    f"Not enough attempts: {len(df)} < {self.config.min_attempts}"
                )
                return False
            logger.info(f"Attempts validation passed: {len(df)} attempts")
            return True

        except Exception as e:
            raise CustomException(e, sys)

    def validate_no_nulls(self, df: pd.DataFrame) -> bool:
        try:
            nulls = df[list(self.config.required_columns)].isnull().sum().sum()
            if nulls > 0:
                logger.error(f"Found {nulls} null values in required columns")
                return False
            logger.info("Null check passed")
            return True

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(
        self, artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        try:
            logger.info("Starting data validation")

            df = pd.read_csv(artifact.train_data_path)

            checks = {
                "columns"      : self.validate_columns(df),
                "min_users"    : self.validate_min_users(df),
                "min_attempts" : self.validate_min_attempts(df),
                "no_nulls"     : self.validate_no_nulls(df),
            }

            all_passed = all(checks.values())

            if all_passed:
                msg = "All validation checks passed"
            else:
                failed = [k for k, v in checks.items() if not v]
                msg    = f"Validation failed for: {failed}"

            logger.info(msg)

            return DataValidationArtifact(
                validation_status = all_passed,
                message           = msg
            )

        except Exception as e:
            raise CustomException(e, sys)