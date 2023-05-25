import pickle
import numpy as np


class Runner:
    def __init__(self, model_path):
        # Load checkpoint from model.pkt
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def run(self, data):
        predictions = self.model.predict_proba(data).round(3)
        activities = [
            "Sitzen",
            "Laufen",
            "Velofahren",
            "Rennen",
            "Stehen",
            "Treppenlaufen",
        ]

        results = {}
        for i, prediction in enumerate(predictions):
            results[i] = {activities[j]: prediction[j] for j in range(len(activities))}

        return results
