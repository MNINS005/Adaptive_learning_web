import os
import sys
import json
import pickle
import numpy as np

from src.logger import logger
from src.exception import CustomException
from src.entity.config_entity import RLTrainerConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DKTTrainerArtifact,
    RLTrainerArtifact
)


class ContextualBandit:
    """
    Simple 2-layer policy network that maps
    (state + question_features) → expected reward
    """

    def __init__(self, state_dim: int, hidden_dim: int = 64):
        self.state_dim  = state_dim
        self.hidden_dim = hidden_dim

        # initialize weights randomly
        self.W1 = np.random.randn(state_dim + 2, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * 0.01
        self.b2 = np.zeros(1)

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, state: np.ndarray,
                difficulty: float,
                topic_idx: int) -> float:
        # concatenate state with question features
        x  = np.concatenate([state, [difficulty, float(topic_idx)]])
        h  = self.relu(x @ self.W1 + self.b1)
        out = self.sigmoid(h @ self.W2 + self.b2)
        return float(out[0])

    def update(self, state: np.ndarray, difficulty: float,
               topic_idx: int, reward: float, lr: float = 0.01):
        # simple gradient step
        x        = np.concatenate([state, [difficulty, float(topic_idx)]])
        h        = self.relu(x @ self.W1 + self.b1)
        pred     = self.sigmoid(h @ self.W2 + self.b2)[0]
        error    = reward - pred

        # backprop
        d_out    = error * pred * (1 - pred)
        self.W2 += lr * h.reshape(-1, 1) * d_out
        self.b2 += lr * d_out

        d_h      = (d_out * self.W2.T).flatten()
        d_h     *= (h > 0).astype(float)
        self.W1 += lr * x.reshape(-1, 1) * d_h
        self.b1 += lr * d_h


class RLTrainer:

    def __init__(self, config: RLTrainerConfig = RLTrainerConfig()):
        self.config = config

    # ── compute ZPD reward ─────────────────────────────────────────────
    def compute_reward(self, is_correct: bool,
                       q_difficulty: float,
                       user_skill: float) -> float:
        if is_correct:
            reward  = 1.0
            reward += max(0, q_difficulty - user_skill) * 0.5
        else:
            reward  = -0.3
            reward += max(0, q_difficulty - user_skill) * 0.2

        # penalty for too easy
        if q_difficulty < user_skill - 0.3:
            reward -= 0.5

        # penalty for too hard
        if q_difficulty > user_skill + 0.4:
            reward -= 0.2

        return float(reward)

    # ── epsilon greedy action selection ───────────────────────────────
    def select_action(self, agent: ContextualBandit,
                      state: np.ndarray,
                      candidates: list,
                      topic_map: dict,
                      epsilon: float) -> dict:
        if np.random.random() < epsilon:
            # explore — pick random question
            return np.random.choice(candidates)

        # exploit — pick question with highest predicted reward
        scores = []
        for q in candidates:
            topic_idx = topic_map.get(q.get("topic", "arrays"), 0)
            score     = agent.forward(state, q["difficulty"], topic_idx)
            scores.append(score)

        return candidates[int(np.argmax(scores))]

    # ── simulate one episode ───────────────────────────────────────────
    def simulate_episode(self, agent: ContextualBandit,
                         dkt_model,
                         user_history: np.ndarray,
                         questions: list,
                         topic_map: dict,
                         epsilon: float,
                         n_steps: int = 10) -> float:
        total_reward = 0.0
        state        = user_history   # DKT output = knowledge state

        for step in range(n_steps):
            if not questions:
                break

            # select question
            action = self.select_action(
                agent, state, questions, topic_map, epsilon
            )

            # simulate user answer based on difficulty vs state
            user_skill   = float(np.mean(state))
            prob_correct = max(0.1, min(0.9, 0.8 - action["difficulty"] + user_skill))
            is_correct   = np.random.random() < prob_correct

            # compute reward
            reward       = self.compute_reward(
                is_correct, action["difficulty"], user_skill
            )
            total_reward += reward

            # update agent
            topic_idx = topic_map.get(action.get("topic", "arrays"), 0)
            agent.update(state, action["difficulty"], topic_idx, reward,
                         lr=self.config.learning_rate)

            # simulate state update (skill improves slightly if correct)
            if is_correct:
                state = np.clip(state + 0.01, 0, 1)

        return total_reward / n_steps

    # ── main entry point ───────────────────────────────────────────────
    def initiate_rl_training(
        self,
        transformation_artifact: DataTransformationArtifact,
        dkt_artifact: DKTTrainerArtifact
    ) -> RLTrainerArtifact:
        try:
            logger.info("Starting RL training")

            # load DKT model
            import keras
            dkt_model = keras.models.load_model(dkt_artifact.model_path)
            logger.info("DKT model loaded")

            # load question index
            with open(transformation_artifact.encoder_path) as f:
                question_index = json.load(f)

            num_questions = transformation_artifact.num_questions

            # load X_train to get user states
            X_train = np.load(transformation_artifact.transformed_train_path)
            logger.info(f"Loaded {X_train.shape[0]} user sequences")

            # build topic map
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

            # build simple question list for simulation
            questions = []
            difficulties = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            topics       = list(topic_map.keys())
            for d in difficulties:
                for t in topics:
                    questions.append({"difficulty": d, "topic": t})

            # initialize agent
            # state_dim = num_questions (DKT output size)
            agent    = ContextualBandit(
                state_dim  = num_questions,
                hidden_dim = 64
            )

            # training loop
            episodes         = self.config.episodes
            epsilon_start    = 0.3
            epsilon_end      = 0.05
            epsilon_decay    = (epsilon_start - epsilon_end) / episodes
            rewards_history  = []

            logger.info(f"Training RL agent for {episodes} episodes")

            for ep in range(episodes):
                epsilon = max(epsilon_end, epsilon_start - epsilon_decay * ep)

                # pick random user state from training data
                user_idx     = np.random.randint(0, X_train.shape[0])
                user_seq     = X_train[user_idx:user_idx+1]   # (1, seq_len, 2N)

                # get DKT state for this user
                dkt_output   = dkt_model.predict(user_seq, verbose=0)
                # take last timestep output as current knowledge state
                state        = dkt_output[0, -1, :]           # (num_questions,)

                # simulate episode
                avg_reward   = self.simulate_episode(
                    agent, dkt_model, state,
                    questions, topic_map, epsilon
                )
                rewards_history.append(avg_reward)

                if (ep + 1) % 100 == 0:
                    recent_avg = np.mean(rewards_history[-100:])
                    logger.info(f"Episode {ep+1}/{episodes} | avg_reward: {recent_avg:.4f} | epsilon: {epsilon:.3f}")
                    print(f"Episode {ep+1}/{episodes} | avg_reward: {recent_avg:.4f} | epsilon: {epsilon:.3f}")

            avg_reward = float(np.mean(rewards_history[-100:]))
            logger.info(f"RL training complete | final avg_reward: {avg_reward:.4f}")

            # save agent
            os.makedirs(self.config.policy_dir, exist_ok=True)
            with open(self.config.policy_path, "wb") as f:
                pickle.dump(agent, f)
            logger.info(f"RL policy saved: {self.config.policy_path}")

            return RLTrainerArtifact(
                policy_path = self.config.policy_path,
                avg_reward  = round(avg_reward, 4),
                episodes    = episodes
            )

        except Exception as e:
            raise CustomException(e, sys)