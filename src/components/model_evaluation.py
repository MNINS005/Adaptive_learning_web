import os
import sys
import json
import pickle
import numpy as np
import mlflow
import mlflow.keras

from src.logger import logger
from src.exception import CustomException
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DKTTrainerArtifact,
    RLTrainerArtifact,
    ModelEvaluationArtifact
)


class ModelEvaluation:

    def __init__(self, config: ModelEvaluationConfig = ModelEvaluationConfig()):
        self.config = config

    # ── evaluate DKT model ─────────────────────────────────────────────
    def evaluate_dkt(self, dkt_artifact: DKTTrainerArtifact,
                     trans_artifact: DataTransformationArtifact) -> dict:
        try:
            logger.info("Evaluating DKT model")

            import keras
            model   = keras.models.load_model(dkt_artifact.model_path)

            X_test  = np.load(trans_artifact.transformed_test_path)
            y_test  = np.load(
                trans_artifact.transformed_test_path.replace("X_test", "y_test")
            )

            results = model.evaluate(X_test, y_test, verbose=0)

            metrics = {
                "test_loss" : round(float(results[0]), 4),
                "test_auc"  : round(float(results[1]), 4),
            }

            logger.info(f"DKT evaluation: {metrics}")
            return metrics

        except Exception as e:
            raise CustomException(e, sys)

    # ── evaluate RL agent ──────────────────────────────────────────────
    def evaluate_rl(self, rl_artifact: RLTrainerArtifact,
                    dkt_artifact: DKTTrainerArtifact,
                    trans_artifact: DataTransformationArtifact) -> dict:
        try:
            logger.info("Evaluating RL agent")

            # load agent
            with open(rl_artifact.policy_path, "rb") as f:
                agent = pickle.load(f)

            # load DKT model
            import keras
            dkt_model = keras.models.load_model(dkt_artifact.model_path)

            # load test data
            X_test = np.load(trans_artifact.transformed_test_path)

            # simulate episodes on test users
            topic_map = {
                "arrays":              0,
                "linked_lists":        1,
                "trees":               2,
                "graphs":              3,
                "dynamic_programming": 4,
                "sorting":             5,
                "searching":           6,
                "recursion":           7,
                "general_cs":          8,
            }

            questions = []
            for d in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                for t in topic_map.keys():
                    questions.append({"difficulty": d, "topic": t})

            rewards       = []
            correct_rates = []

            for i in range(X_test.shape[0]):
                user_seq   = X_test[i:i+1]
                dkt_output = dkt_model.predict(user_seq, verbose=0)
                state      = dkt_output[0, -1, :]
                user_skill = float(np.mean(state))

                ep_rewards  = []
                ep_correct  = []

                for _ in range(10):
                    # pick best question (no exploration in eval)
                    scores = []
                    for q in questions:
                        t_idx = topic_map.get(q["topic"], 0)
                        score = agent.forward(state, q["difficulty"], t_idx)
                        scores.append(score)

                    best_q       = questions[int(np.argmax(scores))]
                    prob_correct = max(0.1, min(0.9,
                        0.8 - best_q["difficulty"] + user_skill
                    ))
                    is_correct   = np.random.random() < prob_correct

                    # compute reward
                    reward = self._compute_reward(
                        is_correct, best_q["difficulty"], user_skill
                    )
                    ep_rewards.append(reward)
                    ep_correct.append(int(is_correct))

                rewards.append(np.mean(ep_rewards))
                correct_rates.append(np.mean(ep_correct))

            metrics = {
                "avg_reward"   : round(float(np.mean(rewards)),       4),
                "avg_correct"  : round(float(np.mean(correct_rates)), 4),
                "min_reward"   : round(float(np.min(rewards)),        4),
                "max_reward"   : round(float(np.max(rewards)),        4),
            }

            logger.info(f"RL evaluation: {metrics}")
            return metrics

        except Exception as e:
            raise CustomException(e, sys)

    def _compute_reward(self, is_correct: bool,
                        q_difficulty: float,
                        user_skill: float) -> float:
        if is_correct:
            reward  = 1.0
            reward += max(0, q_difficulty - user_skill) * 0.5
        else:
            reward  = -0.3
            reward += max(0, q_difficulty - user_skill) * 0.2
        if q_difficulty < user_skill - 0.3:
            reward -= 0.5
        if q_difficulty > user_skill + 0.4:
            reward -= 0.2
        return float(reward)

    # ── check if new model beats previous ─────────────────────────────
    def is_model_accepted(self, current_auc: float) -> tuple:
        try:
            client   = mlflow.tracking.MlflowClient()
            runs     = client.search_runs(
                experiment_ids = [
                    mlflow.get_experiment_by_name(
                        self.config.mlflow_experiment
                    ).experiment_id
                ],
                order_by = ["metrics.val_auc DESC"],
                max_results = 2
            )

            # if only one run exists → accept automatically
            if len(runs) <= 1:
                logger.info("First model run — accepting automatically")
                return True, 0.0

            # compare against previous best
            previous_auc = float(
                runs[1].data.metrics.get("val_auc", 0.0)
            )

            accepted = current_auc >= previous_auc - 0.02
            logger.info(
                f"Current AUC: {current_auc} | "
                f"Previous AUC: {previous_auc} | "
                f"Accepted: {accepted}"
            )
            return accepted, previous_auc

        except Exception as e:
            logger.warning(f"Could not compare models: {e} — accepting by default")
            return True, 0.0

    # ── main entry point ───────────────────────────────────────────────
    def initiate_model_evaluation(
        self,
        trans_artifact : DataTransformationArtifact,
        dkt_artifact   : DKTTrainerArtifact,
        rl_artifact    : RLTrainerArtifact
    ) -> ModelEvaluationArtifact:
        try:
            logger.info("Starting model evaluation")

            mlflow.set_experiment(self.config.mlflow_experiment)

            with mlflow.start_run(run_name="model_evaluation"):

                # evaluate DKT
                dkt_metrics = self.evaluate_dkt(dkt_artifact, trans_artifact)

                # evaluate RL
                rl_metrics  = self.evaluate_rl(
                    rl_artifact, dkt_artifact, trans_artifact
                )

                # log all metrics to mlflow
                mlflow.log_metrics({
                    "eval_test_loss"  : dkt_metrics["test_loss"],
                    "eval_test_auc"   : dkt_metrics["test_auc"],
                    "eval_avg_reward" : rl_metrics["avg_reward"],
                    "eval_avg_correct": rl_metrics["avg_correct"],
                })

                # check if model is good enough
                accepted, best_auc = self.is_model_accepted(
                    dkt_metrics["test_auc"]
                )

                if accepted:
                    msg = f"Model accepted — AUC: {dkt_metrics['test_auc']}"
                else:
                    msg = f"Model rejected — AUC {dkt_metrics['test_auc']} below best {best_auc}"

                mlflow.log_param("accepted", accepted)
                mlflow.log_param("message",  msg)

                logger.info(msg)
                print(f"\nEvaluation Summary:")
                print(f"  test_auc     : {dkt_metrics['test_auc']}")
                print(f"  test_loss    : {dkt_metrics['test_loss']}")
                print(f"  avg_reward   : {rl_metrics['avg_reward']}")
                print(f"  avg_correct  : {rl_metrics['avg_correct']}")
                print(f"  accepted     : {accepted}")
                print(f"  message      : {msg}")

            return ModelEvaluationArtifact(
                is_model_accepted = accepted,
                trained_auc       = dkt_metrics["test_auc"],
                best_auc          = best_auc,
                message           = msg
            )

        except Exception as e:
            raise CustomException(e, sys)