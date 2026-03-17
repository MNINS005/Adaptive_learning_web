import os
import sys
import json
import numpy as np
import pandas as pd

from src.logger import logger
from src.exception import CustomException
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact
)


class DataTransformation:

    def __init__(self, config: DataTransformationConfig = DataTransformationConfig()):
        self.config = config

    # ── Step 1: build question index ──────────────────────────────────
    def build_question_index(self, df: pd.DataFrame) -> dict:
        try:
            logger.info("Building question index")

            unique_questions = df["question_id"].unique().tolist()
            question_index   = {qid: idx for idx, qid in enumerate(unique_questions)}

            logger.info(f"Question index built: {len(question_index)} unique questions")
            return question_index

        except Exception as e:
            raise CustomException(e, sys)

    # ── Step 2: encode one attempt into 2N one-hot vector ─────────────
    def encode_attempt(self, question_idx: int,
                       is_correct: int,
                       num_questions: int) -> np.ndarray:
        vec = np.zeros(2 * num_questions, dtype=np.float32)
        if question_idx < num_questions:
            if is_correct:
                vec[question_idx] = 1.0               # correct slot
            else:
                vec[num_questions + question_idx] = 1.0  # incorrect slot
        return vec

    # ── Step 3: build sequences per user ──────────────────────────────
    def build_sequences(self, df: pd.DataFrame,
                        question_index: dict,
                        num_questions: int) -> tuple:
        try:
            logger.info("Building DKT sequences")

            X_list = []   # input sequences
            y_list = []   # target sequences

            # group by user — each user is one sequence
            for user_id, group in df.groupby("user_id"):
                group = group.sort_values("attempted_at").reset_index(drop=True)

                if len(group) < 2:
                    continue   # need at least 2 attempts to make a sequence

                seq_X = []
                seq_y = []

                for _, row in group.iterrows():
                    q_id    = str(row["question_id"])
                    q_idx   = question_index.get(q_id, -1)

                    if q_idx == -1:
                        continue

                    correct = int(row["is_correct"])

                    # input: encode this attempt
                    vec = self.encode_attempt(q_idx, correct, num_questions)
                    seq_X.append(vec)

                    # target: binary vector of correct answers
                    # shift by 1 — predict next question correctness
                    target = np.zeros(num_questions, dtype=np.float32)
                    target[q_idx] = float(correct)
                    seq_y.append(target)

                if len(seq_X) >= 2:
                    X_list.append(np.array(seq_X,  dtype=np.float32))
                    y_list.append(np.array(seq_y,  dtype=np.float32))

            logger.info(f"Built {len(X_list)} user sequences")
            return X_list, y_list

        except Exception as e:
            raise CustomException(e, sys)

    # ── Step 4: pad sequences to same length ──────────────────────────
    def pad_sequences(self, sequences: list,
                      max_len: int = None) -> np.ndarray:
        try:
            if not sequences:
                return np.array([])

            # find max sequence length
            if max_len is None:
                max_len = max(len(s) for s in sequences)

            feat_dim = sequences[0].shape[1]
            padded   = np.zeros(
                (len(sequences), max_len, feat_dim),
                dtype=np.float32
            )

            for i, seq in enumerate(sequences):
                length = min(len(seq), max_len)
                padded[i, :length, :] = seq[:length]

            logger.info(f"Padded sequences: shape={padded.shape}")
            return padded

        except Exception as e:
            raise CustomException(e, sys)

    # ── Step 5: save artifacts ─────────────────────────────────────────
    def save_artifacts(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray,  y_test: np.ndarray,
                       question_index: dict) -> DataTransformationArtifact:
        try:
            logger.info("Saving transformation artifacts")

            save_dir = self.config.transformed_dir
            os.makedirs(save_dir, exist_ok=True)

            # save numpy arrays
            X_train_path = os.path.join(save_dir, "X_train.npy")
            y_train_path = os.path.join(save_dir, "y_train.npy")
            X_test_path  = os.path.join(save_dir, "X_test.npy")
            y_test_path  = os.path.join(save_dir, "y_test.npy")
            index_path   = os.path.join(save_dir, "question_index.json")

            np.save(X_train_path, X_train)
            np.save(y_train_path, y_train)
            np.save(X_test_path,  X_test)
            np.save(y_test_path,  y_test)

            with open(index_path, "w") as f:
                json.dump(question_index, f)

            logger.info(f"X_train shape : {X_train.shape}")
            logger.info(f"y_train shape : {y_train.shape}")
            logger.info(f"X_test shape  : {X_test.shape}")
            logger.info(f"y_test shape  : {y_test.shape}")
            logger.info(f"Artifacts saved to: {save_dir}")

            return DataTransformationArtifact(
                transformed_train_path = X_train_path,
                transformed_test_path  = X_test_path,
                encoder_path           = index_path,
                num_questions          = len(question_index)
            )

        except Exception as e:
            raise CustomException(e, sys)

    # ── main entry point ───────────────────────────────────────────────
    def initiate_data_transformation(
        self,
        ingestion_artifact: DataIngestionArtifact
    ) -> DataTransformationArtifact:
        try:
            logger.info("Starting data transformation")

            # load CSVs
            train_df = pd.read_csv(ingestion_artifact.train_data_path)
            test_df  = pd.read_csv(ingestion_artifact.test_data_path)

            logger.info(f"Train shape: {train_df.shape}")
            logger.info(f"Test shape : {test_df.shape}")

            # build question index from full data
            full_df        = pd.concat([train_df, test_df], ignore_index=True)
            question_index = self.build_question_index(full_df)
            num_questions  = len(question_index)

            # build sequences
            X_train_list, y_train_list = self.build_sequences(
                train_df, question_index, num_questions
            )
            X_test_list,  y_test_list  = self.build_sequences(
                test_df,  question_index, num_questions
            )

            # find global max length for consistent padding
            all_seqs = X_train_list + X_test_list
            max_len  = max(len(s) for s in all_seqs) if all_seqs else 50

            # pad sequences
            X_train = self.pad_sequences(X_train_list, max_len)
            y_train = self.pad_sequences(y_train_list, max_len)
            X_test  = self.pad_sequences(X_test_list,  max_len)
            y_test  = self.pad_sequences(y_test_list,  max_len)

            # save and return artifact
            artifact = self.save_artifacts(
                X_train, y_train,
                X_test,  y_test,
                question_index
            )

            logger.info("Data transformation completed")
            return artifact

        except Exception as e:
            raise CustomException(e, sys)