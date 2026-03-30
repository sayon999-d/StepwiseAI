import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from typing import Dict, List, Tuple, Optional
import os
import math
import inspect

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerRegretModel(nn.Module):
    def __init__(self, input_dim: int = 128, d_model: int = 256, nhead: int = 8,
                 num_encoder_layers: int = 4, dim_feedforward: int = 512,
                 dropout: float = 0.1, num_factors: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.num_factors = num_factors

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout)
        )
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.regret_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.LayerNorm(d_model // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, 1), nn.Sigmoid()
        )
        self.factor_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.factor_head = nn.Sequential(nn.Linear(d_model, num_factors), nn.Softmax(dim=-1))
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.GELU(), nn.Linear(d_model // 4, 1), nn.Sigmoid()
        )
        self.contrastive_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 128)
        )

    def forward(self, x: torch.Tensor, return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        batch_size = x.size(0)
        x = self.input_projection(x)
        x = x.unsqueeze(1)
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1)
        encoded = self.transformer_encoder(x)
        cls_output = encoded[:, 0, :]
        regret_score = self.regret_head(cls_output)
        attn_output, attn_weights = self.factor_attention(encoded, encoded, encoded)
        factor_importance = self.factor_head(attn_output[:, 0, :])
        confidence = self.confidence_head(cls_output)
        result = {
            'regret_score': regret_score, 'factor_importance': factor_importance,
            'confidence': confidence, 'attention_weights': attn_weights
        }
        if return_embeddings:
            embeddings = self.contrastive_head(cls_output)
            result['embeddings'] = F.normalize(embeddings, p=2, dim=1)
        return result

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        similarity = torch.matmul(embeddings, embeddings.T) / self.temperature
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.T).float()
        positive_mask.fill_diagonal_(0)
        exp_sim = torch.exp(similarity)
        exp_sim.fill_diagonal_(0)
        positive_sum = (exp_sim * positive_mask).sum(dim=1)
        total_sum = exp_sim.sum(dim=1)
        loss = -torch.log(positive_sum / (total_sum + 1e-8) + 1e-8)
        mask = positive_mask.sum(dim=1) > 0
        if mask.sum() > 0:
            return loss[mask].mean()
        return torch.tensor(0.0, device=embeddings.device)

class EnhancedFeatureExtractor:
    DECISION_TYPES = ['job_change', 'education', 'skill_investment', 'relocation',
                      'startup', 'freelance', 'promotion', 'career_switch', 'retirement', 'sabbatical']
    FACTOR_CATEGORIES = ['financial', 'work_life_balance', 'career_growth', 'personal_satisfaction',
                         'family', 'health', 'location', 'skills', 'network', 'risk_tolerance']
    INDUSTRIES = ['technology', 'healthcare', 'finance', 'education', 'manufacturing',
                  'retail', 'consulting', 'media', 'government', 'nonprofit', 'other']
    CAREER_STAGES = ['entry', 'early', 'mid', 'senior', 'executive', 'transition']

    def __init__(self):
        self.scaler = StandardScaler()
        self.fitted = False
        self.stage_embeddings = {
            'entry': np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            'early': np.array([0.8, 0.2, 0.0, 0.0, 0.0, 0.8]),
            'mid': np.array([0.5, 0.5, 0.3, 0.0, 0.0, 0.5]),
            'senior': np.array([0.2, 0.7, 0.6, 0.3, 0.0, 0.3]),
            'executive': np.array([0.0, 0.5, 0.8, 0.7, 0.5, 0.2]),
            'transition': np.array([0.5, 0.3, 0.4, 0.4, 0.2, 0.7])
        }

    def extract_features(self, decision_data: Dict) -> np.ndarray:
        features = []
        decision_type = decision_data.get('decision_type', 'job_change')
        type_idx = self.DECISION_TYPES.index(decision_type) if decision_type in self.DECISION_TYPES else 0
        type_onehot = [1.0 if i == type_idx else 0.0 for i in range(len(self.DECISION_TYPES))]
        features.extend(type_onehot)

        industry = decision_data.get('industry', 'other')
        ind_idx = self.INDUSTRIES.index(industry) if industry in self.INDUSTRIES else len(self.INDUSTRIES) - 1
        ind_onehot = [1.0 if i == ind_idx else 0.0 for i in range(len(self.INDUSTRIES))]
        features.extend(ind_onehot)

        stage = self._determine_career_stage(decision_data)
        features.extend(self.stage_embeddings.get(stage, self.stage_embeddings['mid']))

        features.append(decision_data.get('years_experience', 5) / 40.0)
        features.append(decision_data.get('age', 30) / 70.0)
        features.append(decision_data.get('risk_tolerance', 0.5))
        features.append(decision_data.get('financial_stability', 0.5))
        features.append(decision_data.get('family_support', 0.5))
        features.append(decision_data.get('market_conditions', 0.5))
        features.append(decision_data.get('job_security', 0.5))
        features.append(decision_data.get('passion_alignment', 0.5))

        alternatives = decision_data.get('alternatives', [])
        features.append(min(len(alternatives), 10) / 10.0)
        features.append(1.0 if len(alternatives) >= 3 else 0.5)
        features.append(decision_data.get('confidence', 0.5))
        features.append(decision_data.get('time_pressure', 0.5))
        features.append(1.0 - decision_data.get('time_pressure', 0.5))

        factor_weights = decision_data.get('factor_weights', {})
        for category in self.FACTOR_CATEGORIES:
            features.append(factor_weights.get(category, 0.5))

        features.append(min(decision_data.get('previous_regrets', 0), 10) / 10.0)
        features.append(min(decision_data.get('successful_decisions', 0), 10) / 10.0)
        features.append(decision_data.get('decision_frequency', 0.5))
        features.append(decision_data.get('mentor_consultation', 0.0))
        features.append(decision_data.get('peer_opinions', 0.5))
        features.append(decision_data.get('expert_advice', 0.0))
        features.append(decision_data.get('stress_level', 0.5))
        features.append(decision_data.get('excitement_level', 0.5))
        features.append(decision_data.get('fear_of_missing_out', 0.5))
        features.append(decision_data.get('fear_of_failure', 0.5))
        features.append(min(decision_data.get('salary_change_percent', 0) + 50, 100) / 100.0)
        features.append(decision_data.get('savings_runway_months', 6) / 24.0)
        features.append(decision_data.get('financial_obligations', 0.5))

        while len(features) < 128:
            features.append(0.0)
        return np.array(features[:128], dtype=np.float32)

    def _determine_career_stage(self, decision_data: Dict) -> str:
        years = decision_data.get('years_experience', 5)
        age = decision_data.get('age', 30)
        if years < 2: return 'entry'
        elif years < 5: return 'early'
        elif years < 12: return 'mid'
        elif years < 20: return 'senior'
        elif age > 55: return 'transition'
        else: return 'executive'

    def batch_extract(self, decisions: List[Dict]) -> np.ndarray:
        return np.array([self.extract_features(d) for d in decisions])

class ExplainablePredictor:
    def __init__(self, model: nn.Module, feature_extractor: EnhancedFeatureExtractor):
        self.model = model
        self.feature_extractor = feature_extractor
        self.background_data = None
        self.shap_explainer = None

    def setup_explainer(self, background_decisions: List[Dict]):
        try:
            import shap
            self.background_data = np.array([
                self.feature_extractor.extract_features(d) for d in background_decisions
            ])
            def model_predict(x):
                self.model.eval()
                with torch.no_grad():
                    tensor_x = torch.FloatTensor(x)
                    result = self.model(tensor_x)
                    return result['regret_score'].numpy()
            self.shap_explainer = shap.KernelExplainer(model_predict, self.background_data[:50])
        except ImportError:
            self.shap_explainer = None

    def explain(self, decision_data: Dict) -> Dict:
        features = self.feature_extractor.extract_features(decision_data)
        if self.shap_explainer is not None:
            try:
                shap_values = self.shap_explainer.shap_values(features.reshape(1, -1))
                feature_names = self._get_feature_names()
                contributions = list(zip(feature_names, shap_values[0]))
                contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                return {
                    'top_positive_factors': [(n, float(v)) for n, v in contributions if v > 0][:5],
                    'top_negative_factors': [(n, float(v)) for n, v in contributions if v < 0][:5],
                    'shap_values': {n: float(v) for n, v in contributions[:20]}
                }
            except Exception:
                pass
        return self._attention_based_explanation(decision_data)

    def _attention_based_explanation(self, decision_data: Dict) -> Dict:
        features = self.feature_extractor.extract_features(decision_data)
        tensor_x = torch.FloatTensor(features).unsqueeze(0)
        self.model.eval()
        with torch.no_grad():
            result = self.model(tensor_x)
            factor_importance = result['factor_importance'].numpy()[0]
        factor_names = EnhancedFeatureExtractor.FACTOR_CATEGORIES
        contributions = list(zip(factor_names, factor_importance))
        contributions.sort(key=lambda x: x[1], reverse=True)
        return {'top_positive_factors': contributions[:5], 'top_negative_factors': [], 'attention_based': True}

    def _get_feature_names(self) -> List[str]:
        names = []
        for dt in EnhancedFeatureExtractor.DECISION_TYPES:
            names.append(f"decision_type_{dt}")
        for ind in EnhancedFeatureExtractor.INDUSTRIES:
            names.append(f"industry_{ind}")
        for i in range(6):
            names.append(f"career_stage_dim_{i}")
        names.extend(['years_experience', 'age', 'risk_tolerance', 'financial_stability',
                      'family_support', 'market_conditions', 'job_security', 'passion_alignment',
                      'num_alternatives', 'has_enough_alternatives', 'confidence', 'time_pressure', 'time_to_think'])
        for cat in EnhancedFeatureExtractor.FACTOR_CATEGORIES:
            names.append(f"factor_{cat}")
        names.extend(['previous_regrets', 'successful_decisions', 'decision_frequency',
                      'mentor_consultation', 'peer_opinions', 'expert_advice',
                      'stress_level', 'excitement_level', 'fomo', 'fear_of_failure',
                      'salary_change', 'savings_runway', 'financial_obligations'])
        while len(names) < 128:
            names.append(f"feature_{len(names)}")
        return names[:128]

class EnhancedRegretPredictor:
    def __init__(self, model_path: str = "./models", use_gpu: bool = False, auto_init: bool = True):
        self.model_path = model_path
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.feature_extractor = EnhancedFeatureExtractor()
        self.dl_model = TransformerRegretModel().to(self.device)
        self.ml_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        self.contrastive_loss = ContrastiveLoss()
        self.explainer = ExplainablePredictor(self.dl_model, self.feature_extractor)
        self.dl_weight = 0.7
        self.ml_weight = 0.3
        self.is_trained = False
        self.training_history = []
        self.factor_names = EnhancedFeatureExtractor.FACTOR_CATEGORIES

        self._initialize_weights()
        if auto_init:
            self._pretrain_with_synthetic_data()

    def _initialize_weights(self):
        for module in self.dl_model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)

    def _generate_synthetic_training_data(self, n_samples: int = 200) -> List[Tuple[Dict, float]]:
        import random
        random.seed(42)
        np.random.seed(42)

        decision_types = ['job_change', 'career_switch', 'startup', 'education', 'freelance', 'promotion', 'relocation']
        industries = ['technology', 'healthcare', 'finance', 'education', 'consulting']

        base_regret = {
            'job_change': 0.35, 'career_switch': 0.55, 'startup': 0.70,
            'education': 0.25, 'freelance': 0.50, 'promotion': 0.20, 'relocation': 0.45
        }

        training_data = []
        for _ in range(n_samples):
            decision_type = random.choice(decision_types)
            risk_tolerance = random.uniform(0, 1)
            financial_stability = random.uniform(0, 1)
            years_exp = random.randint(0, 30)
            age = random.randint(22, 60)

            regret = base_regret[decision_type]
            regret += (0.5 - risk_tolerance) * 0.3
            regret += (0.5 - financial_stability) * 0.25
            regret += (years_exp - 10) * -0.005
            regret += random.gauss(0, 0.08)
            regret = max(0.05, min(0.95, regret))

            emotions = []
            if regret < 0.4:
                emotions = random.sample(['excited', 'hopeful', 'confident', 'calm'], k=random.randint(1, 3))
            elif regret > 0.6:
                emotions = random.sample(['anxious', 'fearful', 'stressed', 'overwhelmed'], k=random.randint(1, 3))
            else:
                emotions = random.sample(['curious', 'uncertain', 'conflicted', 'motivated'], k=random.randint(1, 2))

            decision = {
                'decision_type': decision_type,
                'description': f'Synthetic {decision_type} decision',
                'risk_tolerance': risk_tolerance,
                'financial_stability': financial_stability,
                'years_experience': years_exp,
                'age': age,
                'industry': random.choice(industries),
                'emotions': emotions,
                'time_pressure': random.uniform(0.2, 0.8),
                'confidence': 1 - abs(regret - 0.5)
            }
            training_data.append((decision, regret))

        return training_data

    def _pretrain_with_synthetic_data(self, epochs: int = 30):
        training_data = self._generate_synthetic_training_data(200)

        X = np.array([self.feature_extractor.extract_features(d) for d, _ in training_data])
        y = np.array([r for _, r in training_data])

        self.ml_model.fit(X, y)

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)

        optimizer = torch.optim.AdamW(self.dl_model.parameters(), lr=0.002, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        criterion = nn.MSELoss()

        self.dl_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            result = self.dl_model(X_tensor)
            loss = criterion(result['regret_score'], y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dl_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        self.is_trained = True
        self.training_history.append({
            'type': 'synthetic_pretrain',
            'epochs': epochs,
            'samples': len(training_data),
            'final_loss': loss.item()
        })

    def predict(self, decision_data: Dict, include_explanation: bool = True) -> Dict:
        features = self.feature_extractor.extract_features(decision_data)
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        self.dl_model.eval()
        with torch.no_grad():
            result = self.dl_model(features_tensor)
            dl_regret = result['regret_score'].cpu().numpy()[0, 0]
            factor_importance = result['factor_importance'].cpu().numpy()[0]
            dl_confidence = result['confidence'].cpu().numpy()[0, 0]

        if self.is_trained:
            ml_regret = self.ml_model.predict(features.reshape(1, -1))[0]
            ml_regret = np.clip(ml_regret, 0, 1)
            predicted_regret = self.dl_weight * dl_regret + self.ml_weight * ml_regret
        else:
            predicted_regret = dl_regret

        risk_tolerance = decision_data.get('risk_tolerance', 0.5)
        financial_stability = decision_data.get('financial_stability', 0.5)
        decision_type = decision_data.get('decision_type', 'job_change')

        type_base_risk = {
            'job_change': 0.35, 'career_switch': 0.55, 'startup': 0.65,
            'education': 0.30, 'freelance': 0.50, 'promotion': 0.25, 'relocation': 0.40
        }
        base_risk = type_base_risk.get(decision_type, 0.4)

        risk_adjustment = (0.5 - risk_tolerance) * 0.25
        financial_adjustment = (0.5 - financial_stability) * 0.20

        emotions = decision_data.get('emotions', [])
        positive_emotions = {'excited', 'hopeful', 'confident', 'motivated', 'curious', 'calm'}
        negative_emotions = {'anxious', 'fearful', 'uncertain', 'overwhelmed', 'stressed', 'conflicted'}
        pos_count = sum(1 for e in emotions if e in positive_emotions)
        neg_count = sum(1 for e in emotions if e in negative_emotions)
        emotion_adjustment = (neg_count - pos_count) * 0.05

        heuristic_regret = base_risk + risk_adjustment + financial_adjustment + emotion_adjustment
        heuristic_regret = np.clip(heuristic_regret, 0.05, 0.95)

        if not self.is_trained:
            predicted_regret = 0.4 * predicted_regret + 0.6 * heuristic_regret
        else:
            predicted_regret = 0.85 * predicted_regret + 0.15 * heuristic_regret

        predicted_regret = float(np.clip(predicted_regret, 0.05, 0.95))

        confidence = self._calculate_confidence(decision_data, dl_confidence)
        top_factors = self._get_top_factors(factor_importance)
        explanation = self.explainer.explain(decision_data) if include_explanation else {}
        recommendations = self._generate_recommendations(decision_data, predicted_regret, top_factors, explanation)

        return {
            "predicted_regret": predicted_regret,
            "confidence": float(confidence),
            "top_factors": top_factors,
            "explanation": explanation,
            "recommendations": recommendations,
            "risk_level": self._get_risk_level(predicted_regret),
            "risk_category": self._get_risk_category(predicted_regret, confidence)
        }

    def train_with_contrastive(self, training_data: List[Tuple[Dict, float]], epochs: int = 100, contrastive_weight: float = 0.3):
        if len(training_data) < 10:
            return {"error": "Insufficient training data"}
        X = np.array([self.feature_extractor.extract_features(d) for d, _ in training_data])
        y = np.array([r for _, r in training_data])
        self.ml_model.fit(X, y)
        y_bins = np.digitize(y, bins=[0.3, 0.6])
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)
        y_bins_tensor = torch.LongTensor(y_bins).to(self.device)
        optimizer = torch.optim.AdamW(self.dl_model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        mse_loss = nn.MSELoss()
        self.dl_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            result = self.dl_model(X_tensor, return_embeddings=True)
            regret_loss = mse_loss(result['regret_score'], y_tensor)
            contrastive = self.contrastive_loss(result['embeddings'], y_bins_tensor)
            total_loss = regret_loss + contrastive_weight * contrastive
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dl_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        self.is_trained = True
        self.explainer.setup_explainer([d for d, _ in training_data[:100]])
        return {"status": "trained", "final_loss": total_loss.item()}

    def update_with_feedback(self, decision_data: Dict, actual_regret: float, learning_rate: float = 0.0005):
        if not hasattr(self, 'feedback_buffer'):
            self.feedback_buffer = []
            self.feedback_count = 0

        features = self.feature_extractor.extract_features(decision_data)
        self.feedback_buffer.append((features, actual_regret))
        self.feedback_count += 1

        if len(self.feedback_buffer) > 100:
            self.feedback_buffer = self.feedback_buffer[-100:]

        if self.feedback_count % 5 == 0 and len(self.feedback_buffer) >= 5:
            return self._batch_update_from_buffer(learning_rate)

        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        target = torch.FloatTensor([[actual_regret]]).to(self.device)

        adaptive_lr = learning_rate * (0.9 ** (self.feedback_count // 50))
        optimizer = torch.optim.AdamW(self.dl_model.parameters(), lr=adaptive_lr, weight_decay=0.01)
        criterion = nn.SmoothL1Loss()

        self.dl_model.train()
        optimizer.zero_grad()
        result = self.dl_model(features_tensor)
        loss = criterion(result['regret_score'], target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dl_model.parameters(), 0.5)
        optimizer.step()

        return {"updated": True, "loss": loss.item(), "buffer_size": len(self.feedback_buffer)}

    def _batch_update_from_buffer(self, learning_rate: float = 0.0005):
        if len(self.feedback_buffer) < 5:
            return {"updated": False, "reason": "insufficient_buffer"}

        import random
        batch_size = min(16, len(self.feedback_buffer))
        batch = random.sample(self.feedback_buffer, batch_size)

        X = np.array([f for f, _ in batch])
        y = np.array([r for _, r in batch])

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)

        optimizer = torch.optim.AdamW(self.dl_model.parameters(), lr=learning_rate, weight_decay=0.01)
        criterion = nn.SmoothL1Loss()

        self.dl_model.train()
        total_loss = 0

        for _ in range(3):
            optimizer.zero_grad()
            result = self.dl_model(X_tensor)
            loss = criterion(result['regret_score'], y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dl_model.parameters(), 0.5)
            optimizer.step()
            total_loss = loss.item()

        self.training_history.append({
            'type': 'feedback_batch',
            'samples': batch_size,
            'loss': total_loss
        })

        return {"updated": True, "batch_loss": total_loss, "buffer_size": len(self.feedback_buffer)}

    def fine_tune(self, training_data: List[Tuple[Dict, float]], epochs: int = 50, patience: int = 5):
        if len(training_data) < 5:
            return {"error": "Need at least 5 samples for fine-tuning"}

        X = np.array([self.feature_extractor.extract_features(d) for d, _ in training_data])
        y = np.array([r for _, r in training_data])

        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        self.ml_model.fit(X_train, y_train)

        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        optimizer = torch.optim.AdamW(self.dl_model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        criterion = nn.SmoothL1Loss()

        best_val_loss = float('inf')
        no_improve = 0
        best_state = None

        self.dl_model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            result = self.dl_model(X_train_t)
            train_loss = criterion(result['regret_score'], y_train_t)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dl_model.parameters(), 1.0)
            optimizer.step()

            self.dl_model.eval()
            with torch.no_grad():
                val_result = self.dl_model(X_val_t)
                val_loss = criterion(val_result['regret_score'], y_val_t).item()
            self.dl_model.train()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.dl_model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state:
            self.dl_model.load_state_dict(best_state)

        self.is_trained = True
        self.training_history.append({
            'type': 'fine_tune',
            'epochs': epoch + 1,
            'samples': len(training_data),
            'best_val_loss': best_val_loss
        })

        return {"status": "fine_tuned", "epochs": epoch + 1, "best_val_loss": best_val_loss}

    def _calculate_confidence(self, decision_data: Dict, model_confidence: float) -> float:
        base_confidence = model_confidence * 0.4
        important_fields = ['years_experience', 'age', 'risk_tolerance', 'financial_stability',
                           'alternatives', 'factor_weights', 'industry', 'confidence']
        completeness = sum(1 for f in important_fields if f in decision_data) / len(important_fields)
        base_confidence += 0.3 * completeness
        if self.is_trained:
            base_confidence += 0.2
        return min(0.95, max(0.3, base_confidence))

    def _get_top_factors(self, factor_importance: np.ndarray) -> List[Tuple[str, float]]:
        indices = np.argsort(factor_importance)[::-1][:5]
        return [(self.factor_names[i], float(factor_importance[i])) for i in indices if i < len(self.factor_names)]

    def _generate_recommendations(self, decision_data: Dict, regret: float,
                                   top_factors: List[Tuple[str, float]], explanation: Dict) -> List[str]:
        recommendations = []
        if regret > 0.7:
            recommendations.append("High regret risk detected. Consider careful reconsideration.")
            recommendations.append("Create a structured pros/cons analysis with weighted factors.")
        elif regret > 0.4:
            recommendations.append("Moderate regret risk. The decision has both promising and concerning aspects.")
        else:
            recommendations.append("Low regret risk. This appears to be a well-aligned decision.")

        factor_recs = {
            "financial": "Create a detailed 12-month financial projection before deciding.",
            "work_life_balance": "Map out a typical week in this new scenario.",
            "career_growth": "Research 3-5 year growth trajectories for this path.",
            "personal_satisfaction": "Journal about your core values and alignment.",
            "family": "Have an in-depth discussion with family members affected.",
            "health": "Consider the stress and health implications.",
            "location": "Visit the new location or environment if possible.",
            "skills": "Identify skill gaps and create a 90-day learning plan.",
            "network": "Connect with people who have made similar transitions.",
            "risk_tolerance": "Define your personal walk-away point before proceeding."
        }
        for factor, importance in top_factors[:3]:
            if importance > 0.12 and factor in factor_recs:
                recommendations.append(factor_recs[factor])

        if 'top_positive_factors' in explanation:
            positives = [f[0] for f in explanation['top_positive_factors'][:2]]
            if positives:
                recommendations.append(f"Leverage your strengths: Focus on {', '.join(positives)}.")
        return recommendations

    def _get_risk_level(self, regret: float) -> str:
        if regret < 0.3: return "low"
        elif regret < 0.6: return "moderate"
        else: return "high"

    def _get_risk_category(self, regret: float, confidence: float) -> str:
        if regret < 0.3 and confidence > 0.7: return "confident_low_risk"
        elif regret < 0.3: return "uncertain_low_risk"
        elif regret < 0.6 and confidence > 0.7: return "confident_moderate_risk"
        elif regret < 0.6: return "uncertain_moderate_risk"
        elif confidence > 0.7: return "confident_high_risk"
        else: return "uncertain_high_risk"

    def save_model(self, path: Optional[str] = None):
        save_path = path or os.path.join(self.model_path, "enhanced_regret_model.pt")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "dl_model_state": self.dl_model.state_dict(),
            "is_trained": self.is_trained,
            "dl_weight": self.dl_weight,
            "ml_weight": self.ml_weight,
            "training_history": self.training_history
        }, save_path)

    def load_model(self, path: Optional[str] = None):
        load_path = path or os.path.join(self.model_path, "enhanced_regret_model.pt")
        if os.path.exists(load_path):
            load_kwargs = {"map_location": self.device}
            if "weights_only" in inspect.signature(torch.load).parameters:
                load_kwargs["weights_only"] = True
            checkpoint = torch.load(load_path, **load_kwargs)
            self.dl_model.load_state_dict(checkpoint["dl_model_state"])
            self.is_trained = checkpoint.get("is_trained", False)
            self.dl_weight = checkpoint.get("dl_weight", 0.7)
            self.ml_weight = checkpoint.get("ml_weight", 0.3)
            self.training_history = checkpoint.get("training_history", [])

RegretPredictor = EnhancedRegretPredictor
FeatureExtractor = EnhancedFeatureExtractor
