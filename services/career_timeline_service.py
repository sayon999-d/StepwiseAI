import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum


class MilestoneType(Enum):
    DECISION = "decision"
    OUTCOME = "outcome"
    ACHIEVEMENT = "achievement"
    SKILL_ACQUIRED = "skill_acquired"
    SALARY_CHANGE = "salary_change"
    ROLE_CHANGE = "role_change"
    GOAL_COMPLETED = "goal_completed"
    COACHING_SESSION = "coaching_session"
    CUSTOM = "custom"


class ChapterPhase(Enum):
    EXPLORATION = "exploration"
    GROWTH = "growth"
    TRANSITION = "transition"
    MASTERY = "mastery"
    REINVENTION = "reinvention"


@dataclass
class TimelineMilestone:
    id: str
    user_id: str
    milestone_type: MilestoneType
    title: str
    description: str
    date: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    impact_score: float = 0.5 
    tags: List[str] = field(default_factory=list)


@dataclass
class CareerChapter:
    id: str
    user_id: str
    title: str
    phase: ChapterPhase
    start_date: datetime
    end_date: Optional[datetime]
    narrative: str
    key_decisions: List[str]
    metrics: Dict[str, Any]
    milestones: List[str]


@dataclass
class CareerMetricSnapshot:
    date: datetime
    satisfaction: float
    salary_index: float
    skill_count: int
    decision_quality: float
    goals_completed: int


class CareerTimelineService:
    def __init__(self):
        self.milestones: Dict[str, List[TimelineMilestone]] = defaultdict(list)
        self.chapters: Dict[str, List[CareerChapter]] = defaultdict(list)
        self.metrics_history: Dict[str, List[CareerMetricSnapshot]] = defaultdict(list)
        self.user_metadata: Dict[str, Dict] = {}

    def add_milestone(self, user_id: str, milestone_type: str,
                      title: str, description: str, date: str = None,
                      metadata: Dict = None, impact_score: float = 0.5,
                      tags: List[str] = None) -> Dict:
        try:
            mtype = MilestoneType(milestone_type)
        except ValueError:
            mtype = MilestoneType.CUSTOM

        milestone_date = datetime.fromisoformat(date) if date else datetime.utcnow()

        milestone = TimelineMilestone(
            id=str(uuid.uuid4()),
            user_id=user_id,
            milestone_type=mtype,
            title=title,
            description=description,
            date=milestone_date,
            metadata=metadata or {},
            impact_score=max(0, min(1, impact_score)),
            tags=tags or []
        )
        self.milestones[user_id].append(milestone)
        self.milestones[user_id].sort(key=lambda m: m.date)

        return {
            "milestone_id": milestone.id,
            "type": mtype.value,
            "title": title,
            "date": milestone_date.isoformat(),
            "total_milestones": len(self.milestones[user_id])
        }

    def record_metric_snapshot(self, user_id: str, satisfaction: float = None,
                                salary_index: float = None, skill_count: int = None,
                                decision_quality: float = None,
                                goals_completed: int = None) -> Dict:
        history = self.metrics_history[user_id]

        last = history[-1] if history else CareerMetricSnapshot(
            date=datetime.utcnow(), satisfaction=50, salary_index=100,
            skill_count=0, decision_quality=50, goals_completed=0
        )

        snapshot = CareerMetricSnapshot(
            date=datetime.utcnow(),
            satisfaction=satisfaction if satisfaction is not None else last.satisfaction,
            salary_index=salary_index if salary_index is not None else last.salary_index,
            skill_count=skill_count if skill_count is not None else last.skill_count,
            decision_quality=decision_quality if decision_quality is not None else last.decision_quality,
            goals_completed=goals_completed if goals_completed is not None else last.goals_completed
        )
        history.append(snapshot)

        return {
            "date": snapshot.date.isoformat(),
            "satisfaction": snapshot.satisfaction,
            "salary_index": snapshot.salary_index,
            "skill_count": snapshot.skill_count,
            "decision_quality": snapshot.decision_quality
        }

    def get_timeline(self, user_id: str, start_date: str = None,
                     end_date: str = None, types: List[str] = None) -> Dict:
        user_milestones = self.milestones.get(user_id, [])

        if start_date:
            sd = datetime.fromisoformat(start_date)
            user_milestones = [m for m in user_milestones if m.date >= sd]
        if end_date:
            ed = datetime.fromisoformat(end_date)
            user_milestones = [m for m in user_milestones if m.date <= ed]
        if types:
            user_milestones = [m for m in user_milestones if m.milestone_type.value in types]

        metrics = self.metrics_history.get(user_id, [])
        metrics_trend = [{
            "date": m.date.isoformat(),
            "satisfaction": m.satisfaction,
            "salary_index": m.salary_index,
            "skill_count": m.skill_count,
            "decision_quality": m.decision_quality,
            "goals_completed": m.goals_completed
        } for m in metrics]

        chapters = self._get_or_generate_chapters(user_id)

        total = len(self.milestones.get(user_id, []))
        type_counts = defaultdict(int)
        for m in self.milestones.get(user_id, []):
            type_counts[m.milestone_type.value] += 1

        return {
            "user_id": user_id,
            "total_milestones": total,
            "milestones": [{
                "id": m.id,
                "type": m.milestone_type.value,
                "title": m.title,
                "description": m.description,
                "date": m.date.isoformat(),
                "impact_score": m.impact_score,
                "tags": m.tags,
                "metadata": m.metadata
            } for m in user_milestones],
            "metrics_trend": metrics_trend,
            "chapters": [{
                "id": c.id,
                "title": c.title,
                "phase": c.phase.value,
                "start_date": c.start_date.isoformat(),
                "end_date": c.end_date.isoformat() if c.end_date else None,
                "narrative": c.narrative,
                "key_decisions": c.key_decisions,
                "metrics": c.metrics
            } for c in chapters],
            "type_distribution": dict(type_counts),
            "career_duration_days": self._career_duration(user_id)
        }

    def _career_duration(self, user_id: str) -> int:
        milestones = self.milestones.get(user_id, [])
        if len(milestones) < 2:
            return 0
        return (milestones[-1].date - milestones[0].date).days

    def _get_or_generate_chapters(self, user_id: str) -> List[CareerChapter]:
        """Generate career chapters from milestones."""
        if user_id in self.chapters and self.chapters[user_id]:
            return self.chapters[user_id]

        milestones = self.milestones.get(user_id, [])
        if len(milestones) < 3:
            return []

        chapters = []
        chunk_size = max(3, min(10, len(milestones) // 3 + 1))

        for i in range(0, len(milestones), chunk_size):
            chunk = milestones[i:i + chunk_size]
            if not chunk:
                continue

            types = [m.milestone_type for m in chunk]
            phase = self._determine_phase(types, i, len(milestones))

            decisions = [m.title for m in chunk if m.milestone_type == MilestoneType.DECISION]
            achievements = [m.title for m in chunk if m.milestone_type == MilestoneType.ACHIEVEMENT]

            impact_avg = sum(m.impact_score for m in chunk) / len(chunk)

            narrative = self._generate_chapter_narrative(
                phase, chunk, decisions, achievements, impact_avg
            )

            chapter = CareerChapter(
                id=str(uuid.uuid4()),
                user_id=user_id,
                title=self._chapter_title(phase, i // chunk_size + 1),
                phase=phase,
                start_date=chunk[0].date,
                end_date=chunk[-1].date if i + chunk_size < len(milestones) else None,
                narrative=narrative,
                key_decisions=decisions[:5],
                metrics={
                    "milestone_count": len(chunk),
                    "avg_impact": round(impact_avg, 2),
                    "decision_count": len(decisions),
                    "achievement_count": len(achievements)
                },
                milestones=[m.id for m in chunk]
            )
            chapters.append(chapter)

        self.chapters[user_id] = chapters
        return chapters

    def _determine_phase(self, types: List[MilestoneType], position: int, total: int) -> ChapterPhase:
        decision_count = types.count(MilestoneType.DECISION)
        skill_count = types.count(MilestoneType.SKILL_ACQUIRED)
        role_changes = types.count(MilestoneType.ROLE_CHANGE)

        if role_changes >= 2:
            return ChapterPhase.TRANSITION
        if skill_count >= 3:
            return ChapterPhase.GROWTH
        if decision_count >= 3:
            return ChapterPhase.EXPLORATION
        if position > total * 0.7:
            return ChapterPhase.MASTERY
        return ChapterPhase.GROWTH

    def _chapter_title(self, phase: ChapterPhase, number: int) -> str:
        titles = {
            ChapterPhase.EXPLORATION: f"Chapter {number}: Exploring New Horizons",
            ChapterPhase.GROWTH: f"Chapter {number}: Building Foundations",
            ChapterPhase.TRANSITION: f"Chapter {number}: A Bold Transition",
            ChapterPhase.MASTERY: f"Chapter {number}: Reaching Mastery",
            ChapterPhase.REINVENTION: f"Chapter {number}: Reinventing the Path"
        }
        return titles.get(phase, f"Chapter {number}")

    def _generate_chapter_narrative(self, phase: ChapterPhase, milestones: List,
                                     decisions: List, achievements: List,
                                     avg_impact: float) -> str:
        duration = (milestones[-1].date - milestones[0].date).days
        months = max(1, duration // 30)

        narratives = {
            ChapterPhase.EXPLORATION: (
                f"Over {months} months, this was a period of exploration and discovery. "
                f"{'Key decisions included: ' + ', '.join(decisions[:2]) + '. ' if decisions else ''}"
                f"This phase was characterized by curiosity and openness to new possibilities."
            ),
            ChapterPhase.GROWTH: (
                f"This {months}-month growth phase saw significant skill building and career development. "
                f"{'Notable achievements: ' + ', '.join(achievements[:2]) + '. ' if achievements else ''}"
                f"{'Important decisions were made: ' + ', '.join(decisions[:2]) + '. ' if decisions else ''}"
                f"The trajectory shows steady upward momentum."
            ),
            ChapterPhase.TRANSITION: (
                f"A transformative {months}-month period of change and adaptation. "
                f"{'Major decisions: ' + ', '.join(decisions[:2]) + '. ' if decisions else ''}"
                f"While transitions bring uncertainty, the evidence suggests calculated risk-taking."
            ),
            ChapterPhase.MASTERY: (
                f"After {months} months, this phase reflects growing expertise and confidence. "
                f"{'Achievements unlocked: ' + ', '.join(achievements[:2]) + '. ' if achievements else ''}"
                f"The pattern shows deepening mastery and leadership emergence."
            ),
            ChapterPhase.REINVENTION: (
                f"This {months}-month chapter marks a deliberate reinvention. "
                f"Breaking from established patterns to forge a new direction. "
                f"{'Key pivot decisions: ' + ', '.join(decisions[:2]) + '. ' if decisions else ''}"
            )
        }
        return narratives.get(phase, f"A {months}-month career chapter with {len(milestones)} milestones.")

    def get_progress_report(self, user_id: str) -> Dict:
        milestones = self.milestones.get(user_id, [])
        metrics = self.metrics_history.get(user_id, [])

        if not milestones:
            return {
                "user_id": user_id,
                "message": "No milestones recorded yet. Start tracking your career journey!",
                "total_milestones": 0
            }

        satisfaction_trend = "stable"
        if len(metrics) >= 2:
            first_half = sum(m.satisfaction for m in metrics[:len(metrics) // 2]) / max(len(metrics) // 2, 1)
            second_half = sum(m.satisfaction for m in metrics[len(metrics) // 2:]) / max(len(metrics) - len(metrics) // 2, 1)
            if second_half > first_half + 5:
                satisfaction_trend = "improving"
            elif second_half < first_half - 5:
                satisfaction_trend = "declining"

        latest_metrics = metrics[-1] if metrics else None

        return {
            "user_id": user_id,
            "career_duration_days": self._career_duration(user_id),
            "total_milestones": len(milestones),
            "total_chapters": len(self.chapters.get(user_id, [])),
            "current_metrics": {
                "satisfaction": latest_metrics.satisfaction if latest_metrics else None,
                "salary_index": latest_metrics.salary_index if latest_metrics else None,
                "skill_count": latest_metrics.skill_count if latest_metrics else None,
                "decision_quality": latest_metrics.decision_quality if latest_metrics else None
            },
            "trends": {
                "satisfaction": satisfaction_trend,
                "milestones_per_month": round(len(milestones) / max(self._career_duration(user_id) / 30, 1), 1)
            },
            "recent_milestones": [{
                "title": m.title,
                "type": m.milestone_type.value,
                "date": m.date.isoformat()
            } for m in milestones[-5:]],
            "highlight": self._generate_highlight(milestones, metrics)
        }

    def _generate_highlight(self, milestones: List[TimelineMilestone],
                             metrics: List[CareerMetricSnapshot]) -> str:
        total = len(milestones)
        high_impact = sum(1 for m in milestones if m.impact_score >= 0.7)
        decisions = sum(1 for m in milestones if m.milestone_type == MilestoneType.DECISION)
        achievements = sum(1 for m in milestones if m.milestone_type == MilestoneType.ACHIEVEMENT)

        return (
            f"Your career journey includes {total} tracked milestones, "
            f"with {high_impact} high-impact events. "
            f"You've made {decisions} documented decisions and earned {achievements} achievements. "
            f"{'Your satisfaction has been trending positively. ' if metrics and len(metrics) >= 2 and metrics[-1].satisfaction > metrics[0].satisfaction else ''}"
            f"Keep tracking to unlock deeper insights into your career patterns."
        )

    def export_timeline(self, user_id: str) -> Dict:
        timeline = self.get_timeline(user_id)
        report = self.get_progress_report(user_id)
        return {
            "title": f"Career Progress Report — {user_id}",
            "generated_at": datetime.utcnow().isoformat(),
            "timeline": timeline,
            "report": report
        }


career_timeline_service = CareerTimelineService()
