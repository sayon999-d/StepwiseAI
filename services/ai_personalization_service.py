import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import statistics

from .database_service import db_service


class AIPersonalizationService:
    
    def __init__(self):
        self.user_preferences: Dict[str, Dict] = {}
        self.user_contexts: Dict[str, List[Dict]] = {}
        self.feedback_scores: Dict[str, List[Dict]] = {}
    
    def get_user_context(self, user_id: str) -> Dict:
        decisions, total = db_service.get_decisions(user_id, limit=50)
        
        decision_types = defaultdict(int)
        avg_regret = []
        emotions_used = defaultdict(int)
        keywords = defaultdict(int)
        
        for d in decisions:
            decision_types[d.get('decision_type', 'general')] += 1
            if d.get('predicted_regret'):
                avg_regret.append(d['predicted_regret'])
            
            for e in d.get('emotions', []):
                if isinstance(e, dict):
                    emotion = e.get('emotion', e.get('dominant_emotion', ''))
                else:
                    emotion = str(e)
                if emotion:
                    emotions_used[emotion] += 1
            
            desc = d.get('description', '') + ' ' + d.get('title', '')
            for word in desc.lower().split():
                if len(word) > 4:
                    keywords[word] += 1
        
        upcoming_events = db_service.get_calendar_events(
            user_id,
            start_date=datetime.utcnow().isoformat(),
            end_date=(datetime.utcnow() + timedelta(days=7)).isoformat()
        )
        
        user = db_service.get_user_by_id(user_id)
        preferences = json.loads(user.get('preferences', '{}')) if user else {}
        
        context = {
            "user_id": user_id,
            "total_decisions": total,
            "decision_types": dict(decision_types),
            "primary_focus": max(decision_types.items(), key=lambda x: x[1])[0] if decision_types else "general",
            "avg_predicted_regret": round(statistics.mean(avg_regret), 1) if avg_regret else 50,
            "common_emotions": dict(sorted(emotions_used.items(), key=lambda x: -x[1])[:5]),
            "top_keywords": [k for k, _ in sorted(keywords.items(), key=lambda x: -x[1])[:10]],
            "upcoming_events_count": len(upcoming_events),
            "preferences": preferences,
            "experience_level": self._calculate_experience_level(total)
        }
        
        self.user_contexts[user_id] = self.user_contexts.get(user_id, [])[-9:] + [context]
        
        return context
    
    def _calculate_experience_level(self, total_decisions: int) -> str:
        if total_decisions < 5:
            return "beginner"
        elif total_decisions < 20:
            return "intermediate"
        elif total_decisions < 50:
            return "experienced"
        else:
            return "expert"
    
    def generate_personalized_prompt(self, user_id: str, user_message: str) -> str:
        """Generate a personalized system prompt based on user context"""
        context = self.get_user_context(user_id)
        
        prompt_parts = [
            "You are an AI career advisor specialized in helping with career decisions.",
            f"This user has made {context['total_decisions']} decisions with you."
        ]
        
        if context['experience_level'] == 'beginner':
            prompt_parts.append("Be encouraging and explain concepts thoroughly as this is a new user.")
        elif context['experience_level'] == 'expert':
            prompt_parts.append("This is an experienced user - be direct and focus on advanced insights.")
        
        if context['primary_focus'] != 'general':
            prompt_parts.append(f"They primarily deal with {context['primary_focus'].replace('_', ' ')} decisions.")
        
        if context['common_emotions']:
            top_emotion = list(context['common_emotions'].keys())[0]
            prompt_parts.append(f"They often feel {top_emotion} when making decisions - be mindful of this.")
        
        if context['avg_predicted_regret'] > 60:
            prompt_parts.append("This user tends to predict high regret - help them see clearer perspectives.")
        elif context['avg_predicted_regret'] < 30:
            prompt_parts.append("This user is generally confident - help with blind spot awareness.")
        
        if context['upcoming_events_count'] > 0:
            prompt_parts.append(f"They have {context['upcoming_events_count']} upcoming events this week.")
        
        return " ".join(prompt_parts)
    
    def record_feedback(self, user_id: str, message_id: str, feedback_type: str, 
                       rating: int = None, context: Dict = None):
        feedback = {
            "message_id": message_id,
            "feedback_type": feedback_type,
            "rating": rating,
            "context": context,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        db_service.record_ai_feedback(
            user_id=user_id,
            message_id=message_id,
            feedback_type=feedback_type,
            rating=rating,
            comment=None,
            context=context
        )
        
        if user_id not in self.feedback_scores:
            self.feedback_scores[user_id] = []
        self.feedback_scores[user_id].append(feedback)
    
    def get_feedback_summary(self, user_id: str) -> Dict:
        feedbacks = self.feedback_scores.get(user_id, [])
        
        if not feedbacks:
            return {"total": 0, "avg_rating": None, "positive_rate": None}
        
        ratings = [f['rating'] for f in feedbacks if f.get('rating') is not None]
        positive = len([f for f in feedbacks if f.get('feedback_type') == 'positive'])
        negative = len([f for f in feedbacks if f.get('feedback_type') == 'negative'])
        
        return {
            "total": len(feedbacks),
            "avg_rating": round(statistics.mean(ratings), 1) if ratings else None,
            "positive_count": positive,
            "negative_count": negative,
            "positive_rate": round(positive / len(feedbacks) * 100, 1) if feedbacks else 0
        }
    
    def adjust_regret_prediction(self, user_id: str, base_prediction: float) -> float:
        decisions, _ = db_service.get_decisions(user_id, limit=50)
        
        calibration_data = []
        for d in decisions:
            if d.get('predicted_regret') is not None and d.get('actual_regret') is not None:
                calibration_data.append({
                    'predicted': d['predicted_regret'],
                    'actual': d['actual_regret']
                })
        
        if len(calibration_data) < 3:
            return base_prediction
        
        biases = [c['predicted'] - c['actual'] for c in calibration_data]
        avg_bias = statistics.mean(biases)
        
        adjusted = base_prediction - (avg_bias * 0.5)
        
        return max(0, min(100, adjusted))
    
    def get_personalized_suggestions(self, user_id: str, decision_type: str = None) -> List[Dict]:
        context = self.get_user_context(user_id)
        suggestions = []
        
        if context['common_emotions']:
            top_emotion = list(context['common_emotions'].keys())[0]
            emotion_suggestions = {
                'anxious': "Consider practicing mindfulness before making this decision",
                'stressed': "Take a break and return to this decision with a fresh perspective",
                'excited': "Channel your enthusiasm but also consider potential downsides",
                'sad': "Ensure this decision aligns with your long-term goals, not just immediate relief",
                'happy': "Great mindset for decision-making! Trust your positive instincts",
                'neutral': "Good balanced state - focus on the facts and your values"
            }
            if top_emotion in emotion_suggestions:
                suggestions.append({
                    "type": "emotion_based",
                    "title": "Emotional Awareness",
                    "content": emotion_suggestions[top_emotion]
                })
        
        if decision_type:
            type_decisions = [d for d in context.get('recent_decisions', []) 
                            if d.get('decision_type') == decision_type]
            if len(type_decisions) >= 3:
                suggestions.append({
                    "type": "experience_based",
                    "title": "You Have Experience Here",
                    "content": f"You've made {len(type_decisions)} similar decisions before. Consider what worked well."
                })
        
        if context['avg_predicted_regret'] > 50:
            suggestions.append({
                "type": "regret_awareness",
                "title": "Regret Tendency",
                "content": "You tend to predict higher regret. Consider if you're being too cautious."
            })
        
        if context['experience_level'] == 'beginner':
            suggestions.append({
                "type": "getting_started",
                "title": "Building Decision Confidence",
                "content": "Start with smaller decisions to build your decision-making muscle."
            })
        
        return suggestions[:3]
    
    def learn_from_outcome(self, user_id: str, decision_id: str, actual_regret: float):
        decision = db_service.get_decision(user_id, decision_id)
        
        if not decision:
            return
        
        predicted = decision.get('predicted_regret')
        if predicted is None:
            return
        
        error = abs(predicted - actual_regret)
        
        db_service.record_metric(
            user_id=user_id,
            metric_name='prediction_error',
            value=error,
            metadata={
                'decision_id': decision_id,
                'predicted': predicted,
                'actual': actual_regret,
                'decision_type': decision.get('decision_type')
            }
        )
        
        if error > 30:
            db_service.record_metric(
                user_id=user_id,
                metric_name='large_prediction_miss',
                value=error,
                metadata={
                    'decision_id': decision_id,
                    'direction': 'overestimated' if predicted > actual_regret else 'underestimated'
                }
            )
    
    def get_learning_insights(self, user_id: str) -> Dict:
        decisions, _ = db_service.get_decisions(user_id, limit=100)
        
        type_accuracy = defaultdict(lambda: {'predictions': [], 'actuals': []})
        
        for d in decisions:
            if d.get('predicted_regret') is not None and d.get('actual_regret') is not None:
                dtype = d.get('decision_type', 'general')
                type_accuracy[dtype]['predictions'].append(d['predicted_regret'])
                type_accuracy[dtype]['actuals'].append(d['actual_regret'])
        
        accuracy_by_type = []
        for dtype, data in type_accuracy.items():
            if data['predictions']:
                errors = [abs(p - a) for p, a in zip(data['predictions'], data['actuals'])]
                accuracy_by_type.append({
                    'type': dtype,
                    'count': len(data['predictions']),
                    'avg_error': round(statistics.mean(errors), 1),
                    'best_at': statistics.mean(errors) < 15
                })
        
        strengths = [t['type'] for t in accuracy_by_type if t.get('best_at')]
        weaknesses = [t['type'] for t in accuracy_by_type if not t.get('best_at') and t['count'] >= 3]
        
        return {
            "accuracy_by_type": accuracy_by_type,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendation": self._generate_learning_recommendation(strengths, weaknesses)
        }
    
    def _generate_learning_recommendation(self, strengths: List[str], weaknesses: List[str]) -> str:
        if not strengths and not weaknesses:
            return "Keep making decisions and recording outcomes to improve prediction accuracy."
        
        if strengths:
            strength_text = ', '.join(s.replace('_', ' ') for s in strengths[:2])
            rec = f"You're good at predicting regret for {strength_text} decisions. "
        else:
            rec = ""
        
        if weaknesses:
            weakness_text = ', '.join(w.replace('_', ' ') for w in weaknesses[:2])
            rec += f"Consider more time on {weakness_text} decisions as predictions are less accurate."
        
        return rec or "Your predictions are generally well-calibrated. Keep it up!"


ai_personalization_service = AIPersonalizationService()
