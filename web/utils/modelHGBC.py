import torch
import pickle


class Runner:
    def __init__(self, model_path):
        # Load checkpoint from model.pkt
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def run(self, data):
        # Load onehotencode's
        onehotencode = {
            0: "Sitzen",
            1: "Laufen",
            2: "Velofahren",
            3: "Rennen",
            4: "Stehen",
            5: "Treppenlaufen",
        }

        predictions = self.model.predict(data)

        results = {}
        for i, prediction in enumerate(predictions):
            results[i] = onehotencode[prediction]

        return results
