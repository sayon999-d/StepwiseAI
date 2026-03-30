import random
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

@dataclass
class Experiment:
    id: str
    variants: List[str]
    weights: List[float]
    active: bool = True

class ABTestingService:
    def __init__(self):
        self.experiments: Dict[str, Experiment] = {
            "prompt_style": Experiment("prompt_style", ["professional", "mentorship"], [0.5, 0.5]),
            "model_comparison": Experiment("model_comparison", ["ollama", "openai"], [0.8, 0.2])
        }
        self.user_assignments: Dict[str, Dict[str, str]] = {}

    def get_variant(self, user_id: str, experiment_id: str) -> str:
        if experiment_id not in self.experiments or not self.experiments[experiment_id].active:
            return "control"

        if user_id not in self.user_assignments:
            self.user_assignments[user_id] = {}

        if experiment_id not in self.user_assignments[user_id]:
            exp = self.experiments[experiment_id]
            variant = random.choices(exp.variants, weights=exp.weights, k=1)[0]
            self.user_assignments[user_id][experiment_id] = variant

        return self.user_assignments[user_id][experiment_id]

    def log_conversion(self, user_id: str, experiment_id: str, goal_reached: bool):
        variant = self.get_variant(user_id, experiment_id)
        print(f"User {user_id} in {experiment_id} variant {variant} reached goal: {goal_reached}")

ab_testing_service = ABTestingService()
