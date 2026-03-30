import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import random
import hashlib


class FeedItemType(Enum):
    SALARY_TREND = "salary_trend"
    SKILL_GAP = "skill_gap"
    INDUSTRY_NEWS = "industry_news"
    OPPORTUNITY = "opportunity"
    MARKET_SHIFT = "market_shift"
    LEARNING_RESOURCE = "learning_resource"
    PEER_INSIGHT = "peer_insight"
    ACTION_REMINDER = "action_reminder"


class FeedPriority(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class FeedItem:
    id: str
    item_type: FeedItemType
    title: str
    summary: str
    details: Dict[str, Any]
    priority: FeedPriority
    relevance_score: float
    user_id: str
    read: bool = False
    bookmarked: bool = False
    dismissed: bool = False
    engagement_score: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


@dataclass
class UserFeedPreferences:
    user_id: str
    role: str = "software_engineer"
    industry: str = "technology"
    location: str = "remote_us"
    experience_years: int = 5
    skills: List[str] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)
    frequency: str = "weekly" 
    muted_types: List[str] = field(default_factory=list)
    engagement_history: Dict[str, float] = field(default_factory=dict)


class CareerFeedService:

    SALARY_TEMPLATES = [
        {
            "title": "{role} salaries are {direction} in {location}",
            "summary": "Average {role} compensation has {changed} by {pct}% in {location} over the past quarter. "
                       "The median salary is now ${median:,}.",
            "priority": FeedPriority.HIGH
        },
        {
            "title": "Your skills command a {premium}% salary premium",
            "summary": "Professionals with your skill set ({skills}) are earning {premium}% above the market average "
                       "for {role} positions.",
            "priority": FeedPriority.MEDIUM
        }
    ]

    SKILL_TEMPLATES = [
        {
            "title": "Emerging skill alert: {skill}",
            "summary": "Demand for {skill} expertise has increased {growth}% this quarter. "
                       "Adding this to your skillset could open {opportunities} new role types.",
            "priority": FeedPriority.HIGH
        },
        {
            "title": "Skills gap detected for your career path",
            "summary": "Based on where you're heading, consider building expertise in {missing_skills}. "
                       "{gap_pct}% of {target_role} job postings require these.",
            "priority": FeedPriority.MEDIUM
        }
    ]

    INDUSTRY_TEMPLATES = [
        {
            "title": "{industry} hiring is {trend}",
            "summary": "The {industry} sector shows {trend} hiring activity with {openings:,} open positions. "
                       "Key growth areas: {growth_areas}.",
            "priority": FeedPriority.MEDIUM
        }
    ]

    OPPORTUNITY_TEMPLATES = [
        {
            "title": "New opportunity matches your profile",
            "summary": "A {role} position at a {company_type} company aligns with your experience and career goals. "
                       "Estimated salary range: ${low:,}-${high:,}.",
            "priority": FeedPriority.HIGH
        }
    ]

    PEER_TEMPLATES = [
        {
            "title": "{pct}% of similar professionals made this move",
            "summary": "Among {role}s with {years}+ years of experience, {pct}% recently transitioned to {to_role}. "
                       "Average satisfaction after 6 months: {satisfaction}/100.",
            "priority": FeedPriority.LOW
        }
    ]

    def __init__(self):
        self.user_feeds: Dict[str, List[FeedItem]] = defaultdict(list)
        self.user_preferences: Dict[str, UserFeedPreferences] = {}
        self.engagement_data: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def set_preferences(self, user_id: str, preferences: Dict) -> Dict:
        existing = self.user_preferences.get(user_id, UserFeedPreferences(user_id=user_id))
        for key, value in preferences.items():
            if hasattr(existing, key):
                setattr(existing, key, value)
        self.user_preferences[user_id] = existing
        return {"status": "updated", "preferences": self._prefs_to_dict(existing)}

    def _prefs_to_dict(self, prefs: UserFeedPreferences) -> Dict:
        return {
            "role": prefs.role,
            "industry": prefs.industry,
            "location": prefs.location,
            "experience_years": prefs.experience_years,
            "skills": prefs.skills,
            "interests": prefs.interests,
            "frequency": prefs.frequency,
            "muted_types": prefs.muted_types
        }

    def generate_feed(self, user_id: str, count: int = 10) -> List[Dict]:
        prefs = self.user_preferences.get(user_id, UserFeedPreferences(user_id=user_id))
        items = []

        items.extend(self._generate_salary_items(user_id, prefs))
        items.extend(self._generate_skill_items(user_id, prefs))
        items.extend(self._generate_industry_items(user_id, prefs))
        items.extend(self._generate_opportunity_items(user_id, prefs))
        items.extend(self._generate_peer_items(user_id, prefs))
        items.extend(self._generate_action_items(user_id, prefs))

        items = [i for i in items if i.item_type.value not in prefs.muted_types]

        for item in items:
            item.relevance_score = self._calculate_relevance(item, prefs)

        priority_order = {FeedPriority.HIGH: 3, FeedPriority.MEDIUM: 2, FeedPriority.LOW: 1}
        items.sort(key=lambda x: (priority_order.get(x.priority, 0), x.relevance_score), reverse=True)

        items = items[:count]
        self.user_feeds[user_id] = items

        return [self._item_to_dict(i) for i in items]

    def _generate_salary_items(self, user_id: str, prefs: UserFeedPreferences) -> List[FeedItem]:
        items = []
        role_display = prefs.role.replace('_', ' ').title()
        location_display = prefs.location.replace('_', ' ').title()

        direction = random.choice(['trending up', 'stabilizing', 'growing steadily'])
        changed = 'increased' if 'up' in direction or 'growing' in direction else 'remained stable'
        pct = random.uniform(2, 8)
        base_salaries = {
            'software_engineer': 130000, 'product_manager': 140000,
            'data_scientist': 135000, 'designer': 110000, 'manager': 150000
        }
        median = base_salaries.get(prefs.role, 120000) * (1 + pct / 100)

        items.append(FeedItem(
            id=str(uuid.uuid4()),
            item_type=FeedItemType.SALARY_TREND,
            title=f"{role_display} salaries are {direction} in {location_display}",
            summary=f"Average {role_display} compensation has {changed} by {pct:.1f}% in "
                    f"{location_display} over the past quarter. The median salary is now ${median:,.0f}.",
            details={"direction": direction, "pct_change": round(pct, 1), "median": round(median)},
            priority=FeedPriority.HIGH,
            relevance_score=85,
            user_id=user_id,
            expires_at=datetime.utcnow() + timedelta(days=7)
        ))

        if prefs.skills:
            premium = random.uniform(5, 25)
            skills_display = ', '.join(prefs.skills[:3])
            items.append(FeedItem(
                id=str(uuid.uuid4()),
                item_type=FeedItemType.SALARY_TREND,
                title=f"Your skills command a {premium:.0f}% salary premium",
                summary=f"Professionals with your skill set ({skills_display}) are earning {premium:.0f}% "
                        f"above the market average for {role_display} positions.",
                details={"premium_pct": round(premium, 1), "skills": prefs.skills[:3]},
                priority=FeedPriority.MEDIUM,
                relevance_score=75,
                user_id=user_id,
                expires_at=datetime.utcnow() + timedelta(days=14)
            ))

        return items

    def _generate_skill_items(self, user_id: str, prefs: UserFeedPreferences) -> List[FeedItem]:
        items = []
        emerging_skills = {
            'technology': ['AI Agents', 'Rust', 'WebAssembly', 'Edge Computing', 'MLOps'],
            'finance': ['Blockchain', 'Quantitative Analysis', 'RegTech', 'AI Risk'],
            'healthcare': ['Health AI', 'Telemedicine Platforms', 'Genomics Data', 'FHIR APIs'],
        }
        industry_skills = emerging_skills.get(prefs.industry, emerging_skills['technology'])
        skill = random.choice(industry_skills)
        growth = random.randint(15, 45)
        opportunities = random.randint(5, 20)

        items.append(FeedItem(
            id=str(uuid.uuid4()),
            item_type=FeedItemType.SKILL_GAP,
            title=f"Emerging skill alert: {skill}",
            summary=f"Demand for {skill} expertise has increased {growth}% this quarter. "
                    f"Adding this to your skillset could open {opportunities} new role types.",
            details={"skill": skill, "growth_pct": growth, "new_opportunities": opportunities},
            priority=FeedPriority.HIGH,
            relevance_score=80,
            user_id=user_id,
            expires_at=datetime.utcnow() + timedelta(days=7)
        ))

        return items

    def _generate_industry_items(self, user_id: str, prefs: UserFeedPreferences) -> List[FeedItem]:
        items = []
        industry_display = prefs.industry.replace('_', ' ').title()
        trend = random.choice(['accelerating', 'steady', 'surging', 'recovering'])
        openings = random.randint(5000, 50000)
        growth_areas = random.sample([
            'AI/ML', 'Cloud Infrastructure', 'Cybersecurity', 'Data Engineering',
            'DevOps', 'Platform Engineering', 'Product Analytics'
        ], 3)

        items.append(FeedItem(
            id=str(uuid.uuid4()),
            item_type=FeedItemType.INDUSTRY_NEWS,
            title=f"{industry_display} hiring is {trend}",
            summary=f"The {industry_display} sector shows {trend} hiring activity with {openings:,} "
                    f"open positions. Key growth areas: {', '.join(growth_areas)}.",
            details={"trend": trend, "openings": openings, "growth_areas": growth_areas},
            priority=FeedPriority.MEDIUM,
            relevance_score=70,
            user_id=user_id,
            expires_at=datetime.utcnow() + timedelta(days=7)
        ))

        return items

    def _generate_opportunity_items(self, user_id: str, prefs: UserFeedPreferences) -> List[FeedItem]:
        items = []
        role_display = prefs.role.replace('_', ' ').title()
        company_types = ['fast-growing startup', 'established tech company', 'innovative mid-size firm']
        company = random.choice(company_types)
        base = {'software_engineer': 130000, 'product_manager': 140000,
                'data_scientist': 135000}.get(prefs.role, 120000)
        low = int(base * 0.9)
        high = int(base * 1.3)

        items.append(FeedItem(
            id=str(uuid.uuid4()),
            item_type=FeedItemType.OPPORTUNITY,
            title="New opportunity matches your profile",
            summary=f"A {role_display} position at a {company} aligns with your experience and career goals. "
                    f"Estimated salary range: ${low:,}-${high:,}.",
            details={"role": prefs.role, "company_type": company, "salary_low": low, "salary_high": high},
            priority=FeedPriority.HIGH,
            relevance_score=75,
            user_id=user_id,
            expires_at=datetime.utcnow() + timedelta(days=3)
        ))

        return items

    def _generate_peer_items(self, user_id: str, prefs: UserFeedPreferences) -> List[FeedItem]:
        items = []
        role_display = prefs.role.replace('_', ' ').title()
        transitions = {
            'software_engineer': ['Engineering Manager', 'Staff Engineer', 'Solutions Architect'],
            'product_manager': ['Director of Product', 'VP Product', 'General Manager'],
            'data_scientist': ['ML Engineer', 'Data Engineering Manager', 'AI Research Lead'],
        }
        to_roles = transitions.get(prefs.role, ['Senior ' + role_display, 'Manager'])
        to_role = random.choice(to_roles)
        pct = random.randint(15, 40)
        satisfaction = random.randint(65, 85)

        items.append(FeedItem(
            id=str(uuid.uuid4()),
            item_type=FeedItemType.PEER_INSIGHT,
            title=f"{pct}% of similar professionals made this move",
            summary=f"Among {role_display}s with {prefs.experience_years}+ years of experience, "
                    f"{pct}% recently transitioned to {to_role}. "
                    f"Average satisfaction after 6 months: {satisfaction}/100.",
            details={"from_role": prefs.role, "to_role": to_role,
                     "pct": pct, "satisfaction": satisfaction},
            priority=FeedPriority.LOW,
            relevance_score=60,
            user_id=user_id,
            expires_at=datetime.utcnow() + timedelta(days=14)
        ))

        return items

    def _generate_action_items(self, user_id: str, prefs: UserFeedPreferences) -> List[FeedItem]:
        items = []
        actions = [
            ("Time for a salary benchmark check", "It's been a while since you compared your compensation. "
             "Market data suggests reviewing every 6 months."),
            ("Review your career goals", "A quarterly check-in on your goals helps maintain focus. "
             "Take 15 minutes to assess your progress."),
            ("Update your skills inventory", "Keeping your skills list current helps generate better "
             "insights and recommendations."),
        ]
        title, summary = random.choice(actions)
        items.append(FeedItem(
            id=str(uuid.uuid4()),
            item_type=FeedItemType.ACTION_REMINDER,
            title=title,
            summary=summary,
            details={},
            priority=FeedPriority.LOW,
            relevance_score=55,
            user_id=user_id,
            expires_at=datetime.utcnow() + timedelta(days=7)
        ))
        return items

    def _calculate_relevance(self, item: FeedItem, prefs: UserFeedPreferences) -> float:
        base = item.relevance_score

        type_key = item.item_type.value
        engagement = prefs.engagement_history.get(type_key, 0)
        if engagement > 5:
            base += 10
        elif engagement > 10:
            base += 20

        return min(100, base)

    def mark_read(self, user_id: str, item_id: str) -> Dict:
        for item in self.user_feeds.get(user_id, []):
            if item.id == item_id:
                item.read = True
                self.engagement_data[user_id][item.item_type.value] += 1
                prefs = self.user_preferences.get(user_id, UserFeedPreferences(user_id=user_id))
                prefs.engagement_history[item.item_type.value] = \
                    prefs.engagement_history.get(item.item_type.value, 0) + 1
                return {"status": "read", "item_id": item_id}
        return {"error": "Item not found"}

    def bookmark_item(self, user_id: str, item_id: str) -> Dict:
        for item in self.user_feeds.get(user_id, []):
            if item.id == item_id:
                item.bookmarked = not item.bookmarked
                return {"status": "bookmarked" if item.bookmarked else "unbookmarked",
                        "item_id": item_id}
        return {"error": "Item not found"}

    def dismiss_item(self, user_id: str, item_id: str) -> Dict:
        for item in self.user_feeds.get(user_id, []):
            if item.id == item_id:
                item.dismissed = True
                prefs = self.user_preferences.get(user_id, UserFeedPreferences(user_id=user_id))
                prefs.engagement_history[item.item_type.value] = max(
                    0, prefs.engagement_history.get(item.item_type.value, 0) - 2
                )
                return {"status": "dismissed", "item_id": item_id}
        return {"error": "Item not found"}

    def get_bookmarks(self, user_id: str) -> List[Dict]:
        return [self._item_to_dict(i) for i in self.user_feeds.get(user_id, [])
                if i.bookmarked]

    def get_feed_stats(self, user_id: str) -> Dict:
        items = self.user_feeds.get(user_id, [])
        total = len(items)
        read = sum(1 for i in items if i.read)
        bookmarked = sum(1 for i in items if i.bookmarked)
        dismissed = sum(1 for i in items if i.dismissed)

        type_distribution = defaultdict(int)
        for item in items:
            type_distribution[item.item_type.value] += 1

        return {
            "total_items": total,
            "read": read,
            "bookmarked": bookmarked,
            "dismissed": dismissed,
            "engagement_rate": round((read / max(total, 1)) * 100, 1),
            "type_distribution": dict(type_distribution),
            "engagement_by_type": dict(self.engagement_data.get(user_id, {}))
        }

    def _item_to_dict(self, item: FeedItem) -> Dict:
        return {
            "id": item.id,
            "type": item.item_type.value,
            "title": item.title,
            "summary": item.summary,
            "details": item.details,
            "priority": item.priority.value,
            "relevance_score": round(item.relevance_score, 1),
            "read": item.read,
            "bookmarked": item.bookmarked,
            "dismissed": item.dismissed,
            "created_at": item.created_at.isoformat(),
            "expires_at": item.expires_at.isoformat() if item.expires_at else None
        }


career_feed_service = CareerFeedService()
