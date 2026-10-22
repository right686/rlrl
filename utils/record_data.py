import numpy as np
import csv
import os

from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


#--------------- train中获取 ---------------------
class RecordCallback(BaseCallback):
    def __init__(self, csv_path="logs/train_metrics.csv", window_size=100, verbose=0):
        super().__init__(verbose)
        self.csv_path = csv_path
        self.reward_window = deque(maxlen=window_size)

        self.episode_num = 0
        self.file = None
        self.writer = None

    def _on_training_start(self):
        # 每次训练覆盖原文件；想追加可把 "w" 改成 "a"
        self.file = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["episode", "reward", "mean_reward", "loss"])

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        loss = self.logger.name_to_value.get("train/loss")

        for info in infos:
            episode_info = info.get("episode")
            if episode_info is None:
                continue

            reward = float(episode_info["r"])
            self.reward_window.append(reward)
            mean_reward = float(np.mean(self.reward_window))
            self.episode_num += 1

            self.writer.writerow([
                self.episode_num,
                reward,
                mean_reward,
                "" if loss is None else float(loss),
            ])
            self.file.flush()

        return True

    def _on_training_end(self):
        if self.file is not None:
            self.file.close()
    
def create_train_reward_path(output_record_path):
    os.makedirs(output_record_path, exist_ok=True)
    train_information_path = os.path.join(output_record_path, 'train_information.csv')
    return train_information_path



