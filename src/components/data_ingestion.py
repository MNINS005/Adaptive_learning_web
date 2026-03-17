import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import logger
from src.exception import CustomException
from src.configuration.db_connection import SessionLocal
from src.data_access.attempt_data import AttemptData
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact


class DataIngestion:

    def __init__(self, config: DataIngestionConfig = DataIngestionConfig()):
        self.config = config
        self.db     = SessionLocal()

    # ── fetch all attempts from Postgres ──────────────────────────────
    def fetch_attempts_from_db(self) -> pd.DataFrame:
        try:
            logger.info("Fetching attempts from Postgres")
            ad       = AttemptData(self.db)
            attempts = ad.get_all_attempts_for_training()

            if not attempts:
                raise CustomException("No attempts found in DB", sys)

            records = []
            for a in attempts:
                records.append({
                    "user_id":      str(a.user_id),
                    "question_id":  str(a.question_id),
                    "is_correct":   int(a.is_correct),
                    "time_taken":   a.time_taken   if a.time_taken else 0,
                    "attempted_at": a.attempted_at,
                })

            df = pd.DataFrame(records)
            df = df.sort_values(["user_id", "attempted_at"]).reset_index(drop=True)
            logger.info(f"Fetched {len(df)} attempt records")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    # ── split and save to artifacts ────────────────────────────────────
    def split_and_save(self, df: pd.DataFrame) -> DataIngestionArtifact:
        try:
            logger.info("Splitting into train and test")

            os.makedirs(self.config.artifact_dir, exist_ok=True)

            # split by user so same user doesnt appear in both sets
            unique_users = df["user_id"].unique()
            train_users, test_users = train_test_split(
                unique_users,
                test_size    = self.config.test_size,
                random_state = self.config.random_state
            )

            train_df = df[df["user_id"].isin(train_users)].reset_index(drop=True)
            test_df  = df[df["user_id"].isin(test_users)].reset_index(drop=True)

            train_path = os.path.join(self.config.artifact_dir, "train.csv")
            test_path  = os.path.join(self.config.artifact_dir, "test.csv")

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path,   index=False)

            logger.info(f"Train: {len(train_df)} rows | {len(train_users)} users → {train_path}")
            logger.info(f"Test : {len(test_df)}  rows | {len(test_users)}  users → {test_path}")

            return DataIngestionArtifact(
                train_data_path = train_path,
                test_data_path  = test_path,
                total_records   = len(df)
            )

        except Exception as e:
            raise CustomException(e, sys)
    def fetch_attempts_from_db(self) -> pd.DataFrame:
        try:
            logger.info("Fetching attempts from database")
            ad       = AttemptData(self.db)
            attempts = ad.get_all_attempts_for_training()

            if not attempts:
                logger.warning("No attempts found in DB — returning empty DataFrame")
                return pd.DataFrame(columns=[
                "user_id", "question_id", "is_correct",
                "time_taken", "attempted_at"
                ])

            records = []
            for a in attempts:
                records.append({
                "user_id":      str(a.user_id),
                "question_id":  str(a.question_id),
                "is_correct":   int(a.is_correct),
                "time_taken":   a.time_taken,
                "attempted_at": a.attempted_at,
            })

            df = pd.DataFrame(records)
            logger.info(f"Fetched {len(df)} attempt records")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    # ── main entry point ───────────────────────────────────────────────
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logger.info("Starting data ingestion")
            df = self.fetch_attempts_from_db()

            if df.empty:
                logger.warning("No attempts to ingest yet — skipping split")
                os.makedirs(self.config.artifact_dir, exist_ok=True)
                train_path = os.path.join(self.config.artifact_dir, "train.csv")
                test_path  = os.path.join(self.config.artifact_dir, "test.csv")
                df.to_csv(train_path, index=False)
                df.to_csv(test_path,  index=False)
                return DataIngestionArtifact(
                train_data_path = train_path,
                test_data_path  = test_path,
                total_records   = 0
            )

            artifact = self.split_and_save(df)
            logger.info("Data ingestion completed")
            return artifact

        except Exception as e:
            raise CustomException(e, sys)

        finally:
            self.db.close()