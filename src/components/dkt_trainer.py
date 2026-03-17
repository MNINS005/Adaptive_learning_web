import os
import sys
import json
import numpy as np
import mlflow
import mlflow.keras

from src.logger import logger
from src.exception import CustomException
from src.entity.config_entity import DKTTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DKTTrainerArtifact
)


class DKTTrainer:

    def __init__(self, config: DKTTrainerConfig = DKTTrainerConfig()):
        self.config = config

    # ── build Keras LSTM model ─────────────────────────────────────────
    def build_model(self, num_questions: int):
        try:
            import keras
            from keras import layers, Model

            logger.info(f"Building DKT model: questions={num_questions} hidden={self.config.hidden_size}")

            inp = keras.Input(shape=(None, 2 * num_questions))
            x   = layers.LSTM(
                self.config.hidden_size,
                return_sequences = True,
                dropout          = self.config.dropout
            )(inp)
            x   = layers.LSTM(
                self.config.hidden_size // 2,
                return_sequences = True,
                dropout          = self.config.dropout
            )(x)
            out = layers.Dense(num_questions, activation="sigmoid")(x)

            model = Model(inp, out)
            model.compile(
                optimizer = keras.optimizers.AdamW(
                    learning_rate = self.config.learning_rate
                ),
                loss    = "binary_crossentropy",
                metrics = ["AUC"]
            )

            model.summary()
            return model

        except Exception as e:
            raise CustomException(e, sys)

    # ── load transformed data ──────────────────────────────────────────
    def load_data(self, artifact: DataTransformationArtifact):
        try:
            logger.info("Loading transformed data")

            X_train = np.load(artifact.transformed_train_path)
            X_test  = np.load(artifact.transformed_test_path)

            # load targets
            y_train_path = artifact.transformed_train_path.replace("X_train", "y_train")
            y_test_path  = artifact.transformed_test_path.replace("X_test",  "y_test")

            y_train = np.load(y_train_path)
            y_test  = np.load(y_test_path)

            logger.info(f"X_train: {X_train.shape} | y_train: {y_train.shape}")
            logger.info(f"X_test : {X_test.shape}  | y_test : {y_test.shape}")

            return X_train, y_train, X_test, y_test

        except Exception as e:
            raise CustomException(e, sys)

    # ── train model ────────────────────────────────────────────────────
    def train(self, model, X_train: np.ndarray,
              y_train: np.ndarray, X_val: np.ndarray,
              y_val: np.ndarray):
        try:
            import keras

            logger.info("Starting DKT model training")

            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor              = "val_AUC",
                    patience             = 5,
                    restore_best_weights = True,
                    mode                 = "max"
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor  = "val_loss",
                    factor   = 0.5,
                    patience = 3,
                    verbose  = 1
                )
            ]

            history = model.fit(
                X_train, y_train,
                validation_data = (X_val, y_val),
                epochs          = self.config.epochs,
                batch_size      = self.config.batch_size,
                callbacks       = callbacks,
                verbose         = 1
            )

            return history

        except Exception as e:
            raise CustomException(e, sys)

    # ── save model ─────────────────────────────────────────────────────
    def save_model(self, model) -> str:
        try:
            os.makedirs(self.config.model_dir, exist_ok=True)
            model.save(self.config.model_path)
            logger.info(f"Model saved to: {self.config.model_path}")
            return self.config.model_path

        except Exception as e:
            raise CustomException(e, sys)

    # ── main entry point ───────────────────────────────────────────────
    def initiate_dkt_training(
        self,
        artifact: DataTransformationArtifact
    ) -> DKTTrainerArtifact:
        try:
            logger.info("Starting DKT training pipeline")

            # load data
            X_train, y_train, X_test, y_test = self.load_data(artifact)
            num_questions = artifact.num_questions

            # build model
            model = self.build_model(num_questions)

            # start mlflow run
            mlflow.set_experiment(self.config.mlflow_experiment)

            with mlflow.start_run():

                # log params
                mlflow.log_param("hidden_size",    self.config.hidden_size)
                mlflow.log_param("dropout",        self.config.dropout)
                mlflow.log_param("batch_size",     self.config.batch_size)
                mlflow.log_param("learning_rate",  self.config.learning_rate)
                mlflow.log_param("num_questions",  num_questions)
                mlflow.log_param("train_users",    X_train.shape[0])
                mlflow.log_param("test_users",     X_test.shape[0])

                # train
                history = self.train(
                    model,
                    X_train, y_train,
                    X_test,  y_test
                )

                # get best metrics
                val_auc  = max(history.history.get("val_AUC",  [0]))
                val_loss = min(history.history.get("val_loss", [0]))

                # log metrics
                mlflow.log_metric("val_auc",  val_auc)
                mlflow.log_metric("val_loss", val_loss)

                # log model to mlflow
                mlflow.keras.log_model(model, "dkt_model")

                logger.info(f"Training complete — val_auc: {val_auc:.4f} | val_loss: {val_loss:.4f}")

                # save locally
                model_path = self.save_model(model)

            return DKTTrainerArtifact(
                model_path = model_path,
                val_auc    = round(val_auc,  4),
                val_loss   = round(val_loss, 4)
            )

        except Exception as e:
            raise CustomException(e, sys)