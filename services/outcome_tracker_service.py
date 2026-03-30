import uuid
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum


class FollowUpInterval(Enum):
    THIRTY_DAYS = 30
    NINETY_DAYS = 90
    SIX_MONTHS = 180
    ONE_YEAR = 365


@dataclass
class OutcomeReport:
    id: str
    decision_id: str
    user_id: str
    interval_days: int
    actual_satisfaction: float 
    actual_regret: float 
    salary_delta: Optional[float] = None
    career_growth_score: Optional[float] = None
    work_life_balance: Optional[float] = None
    notes: str = ""
    reported_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PredictionAccuracy:
    decision_id: str
    predicted_regret: float
    actual_regret: float
    error: float
    interval_days: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CalibrationPoint:
    predicted_bucket: str 
    predicted_avg: float
    actual_avg: float
    count: int
    is_calibrated: bool 


@dataclass
class FollowUpReminder:
    id: str
    decision_id: str
    user_id: str
    interval_days: int
    due_date: datetime
    completed: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)


class OutcomeTrackerService:
    def __init__(self):
        self.outcome_reports: Dict[str, List[OutcomeReport]] = defaultdict(list)
        self.predictions: Dict[str, Dict] = {} 
        self.accuracy_records: List[PredictionAccuracy] = []
        self.follow_up_reminders: Dict[str, List[FollowUpReminder]] = defaultdict(list)
        self.user_accuracy_cache: Dict[str, Dict] = {}

    def register_prediction(self, decision_id: str, user_id: str,
                            predicted_regret: float, confidence: float,
                            decision_type: str, metadata: Dict = None):
        self.predictions[decision_id] = {
            "decision_id": decision_id,
            "user_id": user_id,
            "predicted_regret": predicted_regret,
            "confidence": confidence,
            "decision_type": decision_type,
            "metadata": metadata or {},
            "created_at": datetime.utcnow().isoformat()
        }
        self._schedule_follow_ups(decision_id, user_id)
        return {"status": "tracked", "decision_id": decision_id}

    def _schedule_follow_ups(self, decision_id: str, user_id: str):
        now = datetime.utcnow()
        for interval in FollowUpInterval:
            reminder = FollowUpReminder(
                id=str(uuid.uuid4()),
                decision_id=decision_id,
                user_id=user_id,
                interval_days=interval.value,
                due_date=now + timedelta(days=interval.value)
            )
            self.follow_up_reminders[user_id].append(reminder)

    def record_outcome(self, decision_id: str, user_id: str,
                       actual_satisfaction: float, actual_regret: float,
                       salary_delta: float = None,
                       career_growth_score: float = None,
                       work_life_balance: float = None,
                       notes: str = "") -> Dict:
        prediction = self.predictions.get(decision_id)
        if not prediction:
            return {"error": "Decision not found in tracker"}

        created = datetime.fromisoformat(prediction["created_at"])
        days_elapsed = (datetime.utcnow() - created).days
        interval = self._nearest_interval(days_elapsed)

        report = OutcomeReport(
            id=str(uuid.uuid4()),
            decision_id=decision_id,
            user_id=user_id,
            interval_days=interval,
            actual_satisfaction=max(0, min(100, actual_satisfaction)),
            actual_regret=max(0, min(100, actual_regret)),
            salary_delta=salary_delta,
            career_growth_score=career_growth_score,
            work_life_balance=work_life_balance,
            notes=notes
        )
        self.outcome_reports[decision_id].append(report)

        accuracy = PredictionAccuracy(
            decision_id=decision_id,
            predicted_regret=prediction["predicted_regret"],
            actual_regret=actual_regret,
            error=abs(prediction["predicted_regret"] - actual_regret),
            interval_days=interval
        )
        self.accuracy_records.append(accuracy)

        self._complete_follow_up(user_id, decision_id, interval)

        self.user_accuracy_cache.pop(user_id, None)

        return {
            "report_id": report.id,
            "prediction_error": round(accuracy.error, 2),
            "predicted_regret": round(prediction["predicted_regret"], 2),
            "actual_regret": round(actual_regret, 2),
            "interval_days": interval,
            "accuracy_rating": self._error_to_rating(accuracy.error)
        }

    def _nearest_interval(self, days: int) -> int:
        intervals = [i.value for i in FollowUpInterval]
        return min(intervals, key=lambda x: abs(x - days))

    def _complete_follow_up(self, user_id: str, decision_id: str, interval: int):
        for reminder in self.follow_up_reminders.get(user_id, []):
            if reminder.decision_id == decision_id and reminder.interval_days == interval:
                reminder.completed = True
                break

    def _error_to_rating(self, error: float) -> str:
        if error <= 10:
            return "excellent"
        elif error <= 20:
            return "good"
        elif error <= 35:
            return "fair"
        return "needs_improvement"

    def get_pending_follow_ups(self, user_id: str) -> List[Dict]:
        now = datetime.utcnow()
        pending = []
        for reminder in self.follow_up_reminders.get(user_id, []):
            if not reminder.completed and reminder.due_date <= now:
                prediction = self.predictions.get(reminder.decision_id, {})
                pending.append({
                    "reminder_id": reminder.id,
                    "decision_id": reminder.decision_id,
                    "interval_days": reminder.interval_days,
                    "due_date": reminder.due_date.isoformat(),
                    "days_overdue": (now - reminder.due_date).days,
                    "decision_type": prediction.get("decision_type", "unknown"),
                    "original_prediction": round(prediction.get("predicted_regret", 0), 2)
                })
        return sorted(pending, key=lambda x: x["days_overdue"], reverse=True)

    def get_upcoming_follow_ups(self, user_id: str, days_ahead: int = 14) -> List[Dict]:
        now = datetime.utcnow()
        cutoff = now + timedelta(days=days_ahead)
        upcoming = []
        for reminder in self.follow_up_reminders.get(user_id, []):
            if not reminder.completed and now < reminder.due_date <= cutoff:
                prediction = self.predictions.get(reminder.decision_id, {})
                upcoming.append({
                    "reminder_id": reminder.id,
                    "decision_id": reminder.decision_id,
                    "interval_days": reminder.interval_days,
                    "due_date": reminder.due_date.isoformat(),
                    "days_until_due": (reminder.due_date - now).days,
                    "decision_type": prediction.get("decision_type", "unknown")
                })
        return sorted(upcoming, key=lambda x: x["days_until_due"])

    def get_calibration_curve(self, user_id: str = None) -> List[Dict]:
        records = self.accuracy_records
        if user_id:
            tracked_decisions = {did for did, p in self.predictions.items()
                                 if p["user_id"] == user_id}
            records = [r for r in records if r.decision_id in tracked_decisions]

        if not records:
            return self._default_calibration()

        buckets = defaultdict(list)
        for record in records:
            bucket = int(record.predicted_regret // 10) * 10
            bucket = min(bucket, 90)
            buckets[bucket].append(record.actual_regret)

        calibration = []
        for bucket in range(0, 100, 10):
            actual_values = buckets.get(bucket, [])
            if actual_values:
                avg_actual = sum(actual_values) / len(actual_values)
                avg_predicted = bucket + 5 
                calibration.append({
                    "bucket": f"{bucket}-{bucket + 10}%",
                    "predicted_avg": round(avg_predicted, 1),
                    "actual_avg": round(avg_actual, 1),
                    "count": len(actual_values),
                    "is_calibrated": abs(avg_predicted - avg_actual) <= 10
                })

        return calibration

    def _default_calibration(self) -> List[Dict]:
        return [{
            "bucket": f"{b}-{b + 10}%",
            "predicted_avg": b + 5,
            "actual_avg": b + 5,
            "count": 0,
            "is_calibrated": True
        } for b in range(0, 100, 10)]

    def get_accuracy_dashboard(self, user_id: str) -> Dict:
        if user_id in self.user_accuracy_cache:
            cache = self.user_accuracy_cache[user_id]
            if (datetime.utcnow() - datetime.fromisoformat(cache["cached_at"])).seconds < 300:
                return cache["data"]

        tracked_decisions = {did for did, p in self.predictions.items()
                             if p["user_id"] == user_id}
        user_records = [r for r in self.accuracy_records
                        if r.decision_id in tracked_decisions]

        if not user_records:
            dashboard = {
                "total_tracked": len(tracked_decisions),
                "outcomes_recorded": 0,
                "overall_accuracy": None,
                "mean_absolute_error": None,
                "accuracy_trend": [],
                "by_interval": {},
                "by_accuracy_rating": {"excellent": 0, "good": 0, "fair": 0, "needs_improvement": 0},
                "calibration": self._default_calibration(),
                "best_prediction": None,
                "worst_prediction": None,
                "model_improving": None
            }
        else:
            errors = [r.error for r in user_records]
            mae = sum(errors) / len(errors)

            by_interval = {}
            for interval in FollowUpInterval:
                interval_records = [r for r in user_records if r.interval_days == interval.value]
                if interval_records:
                    ie = [r.error for r in interval_records]
                    by_interval[f"{interval.value}_days"] = {
                        "count": len(interval_records),
                        "mean_error": round(sum(ie) / len(ie), 2),
                        "accuracy_pct": round(100 - sum(ie) / len(ie), 1)
                    }

            ratings = defaultdict(int)
            for r in user_records:
                ratings[self._error_to_rating(r.error)] += 1

            sorted_records = sorted(user_records, key=lambda x: x.error)
            best = sorted_records[0]
            worst = sorted_records[-1]

            recent = sorted(user_records, key=lambda x: x.timestamp)[-10:]
            trend = [{"error": round(r.error, 2), "date": r.timestamp.isoformat()}
                     for r in recent]
            improving = None
            if len(recent) >= 4:
                first_half = sum(r.error for r in recent[:len(recent) // 2]) / (len(recent) // 2)
                second_half = sum(r.error for r in recent[len(recent) // 2:]) / (len(recent) - len(recent) // 2)
                improving = second_half < first_half

            dashboard = {
                "total_tracked": len(tracked_decisions),
                "outcomes_recorded": len(user_records),
                "overall_accuracy": round(100 - mae, 1),
                "mean_absolute_error": round(mae, 2),
                "accuracy_trend": trend,
                "by_interval": by_interval,
                "by_accuracy_rating": dict(ratings),
                "calibration": self.get_calibration_curve(user_id),
                "best_prediction": {
                    "decision_id": best.decision_id,
                    "error": round(best.error, 2),
                    "predicted": round(best.predicted_regret, 2),
                    "actual": round(best.actual_regret, 2)
                },
                "worst_prediction": {
                    "decision_id": worst.decision_id,
                    "error": round(worst.error, 2),
                    "predicted": round(worst.predicted_regret, 2),
                    "actual": round(worst.actual_regret, 2)
                },
                "model_improving": improving
            }

        self.user_accuracy_cache[user_id] = {
            "data": dashboard,
            "cached_at": datetime.utcnow().isoformat()
        }
        return dashboard

    def get_outcome_history(self, decision_id: str) -> List[Dict]:
        reports = self.outcome_reports.get(decision_id, [])
        return [{
            "id": r.id,
            "interval_days": r.interval_days,
            "actual_satisfaction": r.actual_satisfaction,
            "actual_regret": r.actual_regret,
            "salary_delta": r.salary_delta,
            "career_growth_score": r.career_growth_score,
            "work_life_balance": r.work_life_balance,
            "notes": r.notes,
            "reported_at": r.reported_at.isoformat()
        } for r in sorted(reports, key=lambda x: x.interval_days)]

    def get_retraining_data(self) -> List[Dict]:
        data = []
        for decision_id, reports in self.outcome_reports.items():
            prediction = self.predictions.get(decision_id)
            if not prediction:
                continue
            for report in reports:
                data.append({
                    "decision_type": prediction["decision_type"],
                    "predicted_regret": prediction["predicted_regret"],
                    "confidence": prediction["confidence"],
                    "actual_regret": report.actual_regret,
                    "actual_satisfaction": report.actual_satisfaction,
                    "interval_days": report.interval_days,
                    "salary_delta": report.salary_delta,
                    "career_growth": report.career_growth_score,
                    "metadata": prediction.get("metadata", {})
                })
        return data


outcome_tracker_service = OutcomeTrackerService()
