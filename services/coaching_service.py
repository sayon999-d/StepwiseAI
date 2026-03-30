from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import random

class BiasType(str, Enum):
    LOSS_AVERSION = "loss_aversion"
    OVERCONFIDENCE = "overconfidence"
    ANCHORING = "anchoring"
    SUNK_COST = "sunk_cost"
    CONFIRMATION_BIAS = "confirmation_bias"
    STATUS_QUO = "status_quo"
    RECENCY_BIAS = "recency_bias"
    AVAILABILITY_HEURISTIC = "availability_heuristic"
    PLANNING_FALLACY = "planning_fallacy"
    OPTIMISM_BIAS = "optimism_bias"

@dataclass
class BiasDetection:

    bias_type: BiasType
    confidence: float
    evidence: List[str]
    mitigation_tips: List[str]

@dataclass
class StrengthWeakness:

    name: str
    category: str
    score: float
    evidence: str
    improvement_tips: List[str] = field(default_factory=list)

@dataclass
class ActionItem:

    id: str
    title: str
    description: str
    priority: str
    category: str
    due_date: Optional[datetime]
    completed: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class CoachingSession:

    user_id: str
    session_type: str
    insights: List[str]
    action_items: List[ActionItem]
    biases_detected: List[BiasDetection]
    strengths: List[StrengthWeakness]
    weaknesses: List[StrengthWeakness]
    personalized_advice: List[str]
    progress_score: float
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class UserProfile:

    user_id: str
    decision_style: str
    risk_profile: str
    primary_biases: List[BiasType]
    strengths: List[str]
    growth_areas: List[str]
    coaching_history: List[str]
    total_sessions: int = 0
    progress_score: float = 0.5
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_session: Optional[datetime] = None

class CoachingService:


    BIAS_INDICATORS = {
        BiasType.LOSS_AVERSION: {
            'keywords': ['afraid to lose', 'risk', 'lose', 'safe', 'protect', 'security'],
            'patterns': ['focusing more on potential losses than gains'],
            'tips': [
                'Reframe decisions in terms of gains, not just avoiding losses',
                'Consider what you might gain by taking calculated risks',
                'Ask: What would I regret NOT doing?'
            ]
        },
        BiasType.OVERCONFIDENCE: {
            'keywords': ['definitely', 'certainly', 'guaranteed', 'sure', 'no doubt'],
            'patterns': ['underestimating risks', 'overestimating own abilities'],
            'tips': [
                'Seek out disconfirming evidence',
                'Consider what could go wrong',
                'Ask others for honest feedback'
            ]
        },
        BiasType.SUNK_COST: {
            'keywords': ['invested', 'already spent', 'too late', 'years of', 'put in'],
            'patterns': ['focusing on past investments rather than future value'],
            'tips': [
                'Focus on future benefits, not past costs',
                'Ask: Would I make this choice today with fresh eyes?',
                'Consider opportunity cost of continuing'
            ]
        },
        BiasType.STATUS_QUO: {
            'keywords': ['comfortable', 'stable', 'known', 'familiar', 'safe'],
            'patterns': ['preference for current state despite better options'],
            'tips': [
                'Imagine you were starting fresh - would you choose this?',
                'List what you might be missing by staying put',
                'Set a review date to reconsider'
            ]
        },
        BiasType.CONFIRMATION_BIAS: {
            'keywords': ['confirms', 'proves', 'supports my view', 'knew it'],
            'patterns': ['seeking only information that supports existing beliefs'],
            'tips': [
                'Actively seek out opposing viewpoints',
                'Ask: What would change my mind?',
                'Consult someone who disagrees with you'
            ]
        },
        BiasType.RECENCY_BIAS: {
            'keywords': ['just happened', 'recently', 'last week', 'latest'],
            'patterns': ['overweighting recent events'],
            'tips': [
                'Look at longer-term trends and patterns',
                'Consider historical data and context',
                'Wait before making decisions after emotional events'
            ]
        },
        BiasType.PLANNING_FALLACY: {
            'keywords': ['quick', 'easy', 'simple', 'no problem', 'just'],
            'patterns': ['underestimating time and complexity'],
            'tips': [
                'Add 50% buffer to time estimates',
                'Break down tasks into smaller steps',
                'Learn from past project timelines'
            ]
        },
        BiasType.OPTIMISM_BIAS: {
            'keywords': ['best case', 'hopefully', 'should work out', 'lucky'],
            'patterns': ['assuming things will go better than average'],
            'tips': [
                'Plan for multiple scenarios including worst case',
                'Research base rates and typical outcomes',
                'Build contingency plans'
            ]
        }
    }

    DECISION_STYLE_INDICATORS = {
        'analytical': ['analyze', 'data', 'research', 'evidence', 'logic', 'pros cons'],
        'intuitive': ['feel', 'gut', 'instinct', 'sense', 'heart', 'know'],
        'balanced': []
    }

    def __init__(self):
        self.user_profiles: Dict[str, UserProfile] = {}
        self.sessions: Dict[str, CoachingSession] = {}
        self.action_items: Dict[str, List[ActionItem]] = defaultdict(list)
        self._behavioral_logs: Dict[str, List[Dict]] = defaultdict(list)

    def log_behavior(self, user_id: str, behavior_type: str, metadata: Dict):
        """Track behavioral signals (hesitation, revision count, etc.)"""
        self._behavioral_logs[user_id].append({
            'type': behavior_type,
            'timestamp': datetime.utcnow(),
            'metadata': metadata
        })

    def get_or_create_profile(self, user_id: str) -> UserProfile:

        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                decision_style='balanced',
                risk_profile='moderate',
                primary_biases=[],
                strengths=[],
                growth_areas=[],
                coaching_history=[]
            )
        return self.user_profiles[user_id]

    def detect_biases(self, text: str, decision_history: List[Dict] = None) -> List[BiasDetection]:

        text_lower = text.lower()
        detected_biases = []

        for bias_type, indicators in self.BIAS_INDICATORS.items():
            evidence = []

            for keyword in indicators['keywords']:
                if keyword in text_lower:
                    evidence.append(f"Used term: '{keyword}'")

            if evidence:
                confidence = min(0.9, 0.4 + len(evidence) * 0.15)
                detected_biases.append(BiasDetection(
                    bias_type=bias_type,
                    confidence=confidence,
                    evidence=evidence[:3],
                    mitigation_tips=indicators['tips']
                ))

        detected_biases.sort(key=lambda b: b.confidence, reverse=True)
        return detected_biases[:3]

    def distinguish_stress_vs_confusion(self, text_analysis: Dict, behavioral_metadata: Dict) -> str:
        text_complexity = text_analysis.get('complexity_score', 0)
        uncertainty_markers = len(text_analysis.get('uncertainty_markers', []))

        hesitation_count = behavioral_metadata.get('hesitations', 0)

        if uncertainty_markers > 3 and hesitation_count > 2:
            return "confusion - needs clarity"
        elif text_complexity > 0.7:
            return "stress - needs simplification"
        return "normal"

    def determine_intervention_level(self, user_profile: UserProfile, risk_level: float) -> str:
        if risk_level > 0.8:
            return "critical_intervention"
        elif risk_level > 0.5:
            return "guided_reflection"
        return "passive_monitoring"

    def analyze_decision_style(self, texts: List[str]) -> str:

        combined_text = ' '.join(texts).lower()

        analytical_score = sum(1 for kw in self.DECISION_STYLE_INDICATORS['analytical'] if kw in combined_text)
        intuitive_score = sum(1 for kw in self.DECISION_STYLE_INDICATORS['intuitive'] if kw in combined_text)

        if analytical_score > intuitive_score + 2:
            return 'analytical'
        elif intuitive_score > analytical_score + 2:
            return 'intuitive'
        else:
            return 'balanced'

    def identify_strengths_weaknesses(
        self,
        decisions: List[Dict],
        outcomes: List[Dict] = None
    ) -> tuple[List[StrengthWeakness], List[StrengthWeakness]]:

        strengths = []
        weaknesses = []

        if not decisions:
            return strengths, weaknesses

        emotions_used = defaultdict(int)
        decision_types = defaultdict(int)
        avg_confidence = 0

        for d in decisions:
            for emotion in d.get('emotions', []):
                emotions_used[emotion] += 1
            decision_types[d.get('decision_type', 'unknown')] += 1
            avg_confidence += d.get('predicted_confidence', 0.5)

        avg_confidence /= len(decisions)

        if avg_confidence > 0.7:
            strengths.append(StrengthWeakness(
                name="Self-Awareness",
                category="strength",
                score=avg_confidence,
                evidence="Consistently provides detailed context for decisions",
                improvement_tips=[]
            ))

        positive_emotions = ['excited', 'confident', 'hopeful', 'motivated']
        positive_count = sum(emotions_used.get(e, 0) for e in positive_emotions)
        if positive_count > len(decisions) * 0.5:
            strengths.append(StrengthWeakness(
                name="Positive Outlook",
                category="strength",
                score=0.7,
                evidence="Maintains positive emotional state during decisions",
                improvement_tips=[]
            ))

        negative_emotions = ['anxious', 'stressed', 'overwhelmed', 'uncertain']
        negative_count = sum(emotions_used.get(e, 0) for e in negative_emotions)
        if negative_count > len(decisions) * 0.6:
            weaknesses.append(StrengthWeakness(
                name="Decision Anxiety",
                category="weakness",
                score=negative_count / len(decisions),
                evidence="Frequently experiences anxiety around decisions",
                improvement_tips=[
                    "Practice small decisions to build confidence",
                    "Set decision deadlines to avoid overthinking",
                    "Focus on what you can control"
                ]
            ))

        if len(set(decision_types.keys())) == 1 and len(decisions) > 3:
            weaknesses.append(StrengthWeakness(
                name="Narrow Focus",
                category="weakness",
                score=0.6,
                evidence="Decisions concentrated in single area",
                improvement_tips=[
                    "Consider how decisions in other life areas connect",
                    "Broaden perspective by exploring adjacent domains"
                ]
            ))

        return strengths, weaknesses

    def generate_action_items(
        self,
        user_id: str,
        biases: List[BiasDetection],
        weaknesses: List[StrengthWeakness],
        decision_context: Dict = None
    ) -> List[ActionItem]:

        items = []

        for bias in biases[:2]:
            items.append(ActionItem(
                id=f"action_{random.randint(1000, 9999)}",
                title=f"Address {bias.bias_type.value.replace('_', ' ').title()} Bias",
                description=bias.mitigation_tips[0] if bias.mitigation_tips else "Practice awareness of this bias",
                priority="medium" if bias.confidence < 0.7 else "high",
                category="bias_mitigation",
                due_date=datetime.utcnow() + timedelta(days=7)
            ))

        for weakness in weaknesses[:2]:
            if weakness.improvement_tips:
                items.append(ActionItem(
                    id=f"action_{random.randint(1000, 9999)}",
                    title=f"Work on: {weakness.name}",
                    description=weakness.improvement_tips[0],
                    priority="medium",
                    category="skill_development",
                    due_date=datetime.utcnow() + timedelta(days=14)
                ))

        items.append(ActionItem(
            id=f"action_{random.randint(1000, 9999)}",
            title="Weekly Reflection",
            description="Spend 15 minutes reviewing your decisions from the past week",
            priority="low",
            category="habit",
            due_date=datetime.utcnow() + timedelta(days=7)
        ))

        self.action_items[user_id].extend(items)

        return items

    def generate_personalized_advice(
        self,
        profile: UserProfile,
        current_decision: Dict = None
    ) -> List[str]:

        advice = []

        if profile.decision_style == 'analytical':
            advice.append("Your analytical approach is valuable. Just remember that not all factors can be quantified.")
        elif profile.decision_style == 'intuitive':
            advice.append("Your intuition is a strength. Support it with some data validation when possible.")
        else:
            advice.append("Your balanced approach serves you well. Continue weighing both logic and intuition.")

        if profile.risk_profile == 'conservative':
            advice.append("Consider whether your caution might be causing you to miss growth opportunities.")
        elif profile.risk_profile == 'aggressive':
            advice.append("Your willingness to take risks can lead to big wins. Ensure you have contingency plans.")

        if profile.primary_biases:
            top_bias = profile.primary_biases[0]
            tips = self.BIAS_INDICATORS.get(top_bias, {}).get('tips', [])
            if tips:
                advice.append(f"Watch out for {top_bias.value.replace('_', ' ')}: {tips[0]}")

        if profile.progress_score > 0.7:
            advice.append("You're making great progress in your decision-making skills!")
        elif profile.progress_score < 0.4:
            advice.append("Consider establishing a regular decision review practice to build confidence.")

        return advice

    def create_coaching_session(
        self,
        user_id: str,
        session_type: str,
        recent_decisions: List[Dict] = None,
        current_text: str = None
    ) -> CoachingSession:

        profile = self.get_or_create_profile(user_id)

        biases = []
        if current_text:
            biases = self.detect_biases(current_text)

        for bias in biases:
            if bias.bias_type not in profile.primary_biases:
                profile.primary_biases.append(bias.bias_type)
        profile.primary_biases = profile.primary_biases[:5]

        if recent_decisions:
            texts = [d.get('description', '') for d in recent_decisions]
            profile.decision_style = self.analyze_decision_style(texts)

        strengths, weaknesses = self.identify_strengths_weaknesses(recent_decisions or [])

        action_items = self.generate_action_items(user_id, biases, weaknesses)

        advice = self.generate_personalized_advice(profile)

        insights = self._generate_insights(profile, recent_decisions)

        self._update_progress(profile, strengths, weaknesses)

        session = CoachingSession(
            user_id=user_id,
            session_type=session_type,
            insights=insights,
            action_items=action_items,
            biases_detected=biases,
            strengths=strengths,
            weaknesses=weaknesses,
            personalized_advice=advice,
            progress_score=profile.progress_score
        )

        profile.total_sessions += 1
        profile.last_session = datetime.utcnow()
        profile.coaching_history.append(session_type)

        return session

    def _generate_insights(self, profile: UserProfile, decisions: List[Dict] = None) -> List[str]:

        insights = []

        if decisions and len(decisions) >= 3:
            recent_regrets = [d.get('predicted_regret', 0.5) for d in decisions[:5]]
            avg_regret = sum(recent_regrets) / len(recent_regrets)

            intervention_level = self.determine_intervention_level(profile, avg_regret)

            if avg_regret < 0.3:
                insights.append("Your recent decisions show low predicted regret - you seem to be in a good decision-making space.")
            elif avg_regret > 0.6:
                insights.append(f"High risk detected. System triggering {intervention_level.replace('_', ' ')}. Consider slowing down.")

            types = [d.get('decision_type') for d in decisions]
            if types:
                most_common = max(set(types), key=types.count)
                insights.append(f"You've been focusing mainly on {most_common.replace('_', ' ')} decisions recently.")

        insights.append(f"Your decision-making style: {profile.decision_style.title()}")
        insights.append(f"Your risk profile: {profile.risk_profile.title()}")

        return insights

    def _update_progress(
        self,
        profile: UserProfile,
        strengths: List[StrengthWeakness],
        weaknesses: List[StrengthWeakness]
    ):

        strength_score = sum(s.score for s in strengths) / max(1, len(strengths)) if strengths else 0.5
        weakness_penalty = sum(w.score for w in weaknesses) / max(1, len(weaknesses)) if weaknesses else 0

        session_bonus = min(0.2, profile.total_sessions * 0.02)

        profile.progress_score = min(1.0, max(0.1,
            0.3 + strength_score * 0.3 - weakness_penalty * 0.2 + session_bonus
        ))

        profile.strengths = [s.name for s in strengths]
        profile.growth_areas = [w.name for w in weaknesses]

    def get_user_action_items(self, user_id: str, include_completed: bool = False) -> List[ActionItem]:

        items = self.action_items.get(user_id, [])
        if not include_completed:
            items = [i for i in items if not i.completed]
        return items

    def complete_action_item(self, user_id: str, action_id: str) -> bool:

        items = self.action_items.get(user_id, [])
        for item in items:
            if item.id == action_id:
                item.completed = True
                return True
        return False

    def get_weekly_checkin(self, user_id: str) -> Dict[str, Any]:

        profile = self.get_or_create_profile(user_id)
        action_items = self.get_user_action_items(user_id)

        completed_this_week = sum(
            1 for items in self.action_items.get(user_id, [])
            if items.completed and items.created_at > datetime.utcnow() - timedelta(days=7)
        )

        return {
            'progress_score': profile.progress_score,
            'decision_style': profile.decision_style,
            'pending_actions': len(action_items),
            'completed_this_week': completed_this_week,
            'primary_focus': profile.growth_areas[0] if profile.growth_areas else None,
            'encouragement': self._get_encouragement(profile.progress_score),
            'tip_of_the_week': self._get_weekly_tip(profile)
        }

    def _get_encouragement(self, progress: float) -> str:

        if progress > 0.8:
            return "Outstanding progress! You're becoming a confident decision-maker."
        elif progress > 0.6:
            return "Great work! Your decision-making skills are improving steadily."
        elif progress > 0.4:
            return "You're on the right track. Keep practicing mindful decision-making."
        else:
            return "Every decision is a learning opportunity. Take it one step at a time."

    def _get_weekly_tip(self, profile: UserProfile) -> str:

        tips = [
            "Before a big decision, write down your criteria for success.",
            "Sleep on important decisions - your subconscious often finds clarity.",
            "Discuss your decision with someone who has a different perspective.",
            "Set a 'decision deadline' to avoid analysis paralysis.",
            "After deciding, avoid second-guessing. Trust your process.",
            "Keep a decision journal to track patterns over time.",
            "Consider the 10/10/10 rule: How will you feel in 10 minutes, 10 months, 10 years?"
        ]

        if profile.primary_biases:
            bias = profile.primary_biases[0]
            bias_tips = self.BIAS_INDICATORS.get(bias, {}).get('tips', [])
            if bias_tips:
                tips.insert(0, bias_tips[0])

        return random.choice(tips)

    def to_dict(self, session: CoachingSession) -> Dict[str, Any]:

        return {
            'user_id': session.user_id,
            'session_type': session.session_type,
            'insights': session.insights,
            'action_items': [
                {
                    'id': a.id,
                    'title': a.title,
                    'description': a.description,
                    'priority': a.priority,
                    'category': a.category,
                    'due_date': a.due_date.isoformat() if a.due_date else None,
                    'completed': a.completed
                }
                for a in session.action_items
            ],
            'biases_detected': [
                {
                    'type': b.bias_type.value,
                    'confidence': b.confidence,
                    'evidence': b.evidence,
                    'mitigation_tips': b.mitigation_tips
                }
                for b in session.biases_detected
            ],
            'strengths': [
                {
                    'name': s.name,
                    'score': s.score,
                    'evidence': s.evidence
                }
                for s in session.strengths
            ],
            'weaknesses': [
                {
                    'name': w.name,
                    'score': w.score,
                    'evidence': w.evidence,
                    'improvement_tips': w.improvement_tips
                }
                for w in session.weaknesses
            ],
            'personalized_advice': session.personalized_advice,
            'progress_score': session.progress_score,
            'created_at': session.created_at.isoformat()
        }

coaching_service = CoachingService()
