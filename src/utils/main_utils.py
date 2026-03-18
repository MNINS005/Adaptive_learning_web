import os
import sys
import json
import pickle
import numpy as np
import yaml

from src.logger import logger
from src.exception import CustomException


# ── file I/O utils ─────────────────────────────────────────────────────

def save_json(data: dict, path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON saved: {path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_json(path: str) -> dict:
    try:
        with open(path, "r") as f:
            data = json.load(f)
        logger.info(f"JSON loaded: {path}")
        return data
    except Exception as e:
        raise CustomException(e, sys)


def save_pickle(obj, path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Pickle saved: {path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_pickle(path: str):
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Pickle loaded: {path}")
        return obj
    except Exception as e:
        raise CustomException(e, sys)


def save_numpy(array: np.ndarray, path: str):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, array)
        logger.info(f"Numpy saved: {path} | shape: {array.shape}")
    except Exception as e:
        raise CustomException(e, sys)


def load_numpy(path: str) -> np.ndarray:
    try:
        array = np.load(path)
        logger.info(f"Numpy loaded: {path} | shape: {array.shape}")
        return array
    except Exception as e:
        raise CustomException(e, sys)


def load_yaml(path: str) -> dict:
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        logger.info(f"YAML loaded: {path}")
        return data
    except Exception as e:
        raise CustomException(e, sys)


def create_directories(paths: list):
    try:
        for path in paths:
            os.makedirs(path, exist_ok=True)
            logger.info(f"Directory created: {path}")
    except Exception as e:
        raise CustomException(e, sys)


# ── DKT utils ──────────────────────────────────────────────────────────

def encode_attempt(question_idx: int,
                   is_correct: int,
                   num_questions: int) -> np.ndarray:
    try:
        vec = np.zeros(2 * num_questions, dtype=np.float32)
        if question_idx < num_questions:
            if is_correct:
                vec[question_idx] = 1.0
            else:
                vec[num_questions + question_idx] = 1.0
        return vec
    except Exception as e:
        raise CustomException(e, sys)


def encode_user_sequence(attempts: list,
                          question_index: dict,
                          num_questions: int) -> np.ndarray:
    try:
        sequence = []
        for attempt in attempts:
            q_id    = str(attempt["question_id"])
            q_idx   = question_index.get(q_id, -1)
            correct = int(attempt["is_correct"])

            if q_idx == -1:
                continue

            vec = encode_attempt(q_idx, correct, num_questions)
            sequence.append(vec)

        if not sequence:
            return np.zeros((1, 2 * num_questions), dtype=np.float32)

        return np.array(sequence, dtype=np.float32)

    except Exception as e:
        raise CustomException(e, sys)


def get_user_skill(dkt_output: np.ndarray) -> float:
    return float(np.mean(dkt_output))


def filter_candidate_questions(questions: list,
                                user_skill: float,
                                topic: str = None,
                                zpd_delta: float = 0.15) -> list:
    try:
        target     = float(np.clip(user_skill + zpd_delta, 0.0, 1.0))
        min_diff   = max(0.0, target - 0.2)
        max_diff   = min(1.0, target + 0.2)

        candidates = [
            q for q in questions
            if min_diff <= q.difficulty <= max_diff
        ]

        if topic:
            topic_candidates = [q for q in candidates if q.topic == topic]
            candidates = topic_candidates if topic_candidates else candidates

        return candidates

    except Exception as e:
        raise CustomException(e, sys)


# ── RL utils ───────────────────────────────────────────────────────────

def compute_reward(is_correct: bool,
                   q_difficulty: float,
                   user_skill: float) -> float:
    try:
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

    except Exception as e:
        raise CustomException(e, sys)


def get_target_difficulty(user_skill: float,
                           zpd_delta: float = 0.15) -> float:
    return float(np.clip(user_skill + zpd_delta, 0.0, 1.0))


# ── model utils ────────────────────────────────────────────────────────

def load_dkt_model(model_path: str):
    try:
        import keras
        model = keras.models.load_model(model_path)
        logger.info(f"DKT model loaded from: {model_path}")
        return model
    except Exception as e:
        raise CustomException(e, sys)


def load_rl_agent(policy_path: str):
    try:
        agent = load_pickle(policy_path)
        logger.info(f"RL agent loaded from: {policy_path}")
        return agent
    except Exception as e:
        raise CustomException(e, sys)


def get_knowledge_state(dkt_model,
                         user_sequence: np.ndarray) -> np.ndarray:
    try:
        inp        = user_sequence[np.newaxis, :]
        dkt_output = dkt_model.predict(inp, verbose=0)
        state      = dkt_output[0, -1, :]
        return state
    except Exception as e:
        raise CustomException(e, sys)