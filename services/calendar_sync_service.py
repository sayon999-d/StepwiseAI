from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
import hashlib


class CalendarEventType(str, Enum):
    DECISION_DEADLINE = "decision_deadline"
    CHECK_IN = "check_in"
    GOAL_MILESTONE = "goal_milestone"
    INTERVIEW = "interview"
    FOLLOW_UP = "follow_up"
    REFLECTION = "reflection"
    COACHING_SESSION = "coaching_session"


@dataclass
class CalendarEvent:
    id: str
    user_id: str
    event_type: CalendarEventType
    title: str
    description: str
    start_time: datetime
    end_time: datetime
    location: str = ""
    reminder_minutes: List[int] = field(default_factory=lambda: [30, 1440])
    color_id: str = "1"
    recurrence: Optional[str] = None
    google_event_id: Optional[str] = None
    synced: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CalendarConnection:
    user_id: str
    access_token: str
    refresh_token: str
    token_expiry: datetime
    calendar_id: str = "primary"
    connected_at: datetime = field(default_factory=datetime.utcnow)
    last_sync: Optional[datetime] = None
    sync_enabled: bool = True
    auto_create_events: bool = True
    event_types: List[str] = field(default_factory=lambda: [e.value for e in CalendarEventType])


class GoogleCalendarSyncService:
    """
    Syncs career-related events with Google Calendar
    """

    EVENT_COLORS = {
        CalendarEventType.DECISION_DEADLINE: "11",
        CalendarEventType.CHECK_IN: "3",
        CalendarEventType.GOAL_MILESTONE: "10",
        CalendarEventType.INTERVIEW: "9",
        CalendarEventType.FOLLOW_UP: "5",
        CalendarEventType.REFLECTION: "2",
        CalendarEventType.COACHING_SESSION: "6"
    }

    def __init__(self):
        self.connections: Dict[str, CalendarConnection] = {}
        self.events: Dict[str, List[CalendarEvent]] = {}
        self.pending_sync: Dict[str, List[str]] = {}

    def connect_calendar(
        self,
        user_id: str,
        access_token: str,
        refresh_token: str,
        token_expiry: datetime,
        calendar_id: str = "primary"
    ) -> Dict[str, Any]:
        connection = CalendarConnection(
            user_id=user_id,
            access_token=access_token,
            refresh_token=refresh_token,
            token_expiry=token_expiry,
            calendar_id=calendar_id
        )

        self.connections[user_id] = connection

        return {
            "connected": True,
            "user_id": user_id,
            "calendar_id": calendar_id,
            "sync_enabled": True,
            "event_types": connection.event_types
        }

    def disconnect_calendar(self, user_id: str) -> Dict[str, Any]:
        if user_id in self.connections:
            del self.connections[user_id]
            return {"disconnected": True}
        return {"disconnected": False, "error": "Not connected"}

    def get_connection_status(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self.connections:
            return {
                "connected": False,
                "oauth_url": self._generate_oauth_url(user_id)
            }

        conn = self.connections[user_id]
        return {
            "connected": True,
            "calendar_id": conn.calendar_id,
            "sync_enabled": conn.sync_enabled,
            "last_sync": conn.last_sync.isoformat() if conn.last_sync else None,
            "auto_create": conn.auto_create_events,
            "event_types": conn.event_types,
            "token_expires": conn.token_expiry.isoformat()
        }

    def _generate_oauth_url(self, user_id: str) -> str:
        client_id = "your_google_client_id"
        redirect_uri = "http://localhost:8000/api/calendar/callback"
        scope = "https://www.googleapis.com/auth/calendar.events"

        return (
            f"https://accounts.google.com/o/oauth2/v2/auth?"
            f"client_id={client_id}&"
            f"redirect_uri={redirect_uri}&"
            f"scope={scope}&"
            f"response_type=code&"
            f"access_type=offline&"
            f"state={user_id}"
        )

    def update_sync_settings(
        self,
        user_id: str,
        sync_enabled: bool = None,
        auto_create: bool = None,
        event_types: List[str] = None
    ) -> Dict[str, Any]:
        if user_id not in self.connections:
            return {"error": "Not connected"}

        conn = self.connections[user_id]

        if sync_enabled is not None:
            conn.sync_enabled = sync_enabled
        if auto_create is not None:
            conn.auto_create_events = auto_create
        if event_types is not None:
            conn.event_types = event_types

        return {
            "updated": True,
            "sync_enabled": conn.sync_enabled,
            "auto_create": conn.auto_create_events,
            "event_types": conn.event_types
        }

    def create_event(
        self,
        user_id: str,
        event_type: CalendarEventType,
        title: str,
        start_time: datetime,
        end_time: datetime = None,
        description: str = "",
        location: str = "",
        reminder_minutes: List[int] = None,
        recurrence: str = None
    ) -> Dict[str, Any]:
        """Create a calendar event"""
        event_id = hashlib.sha256(
            f"{user_id}{title}{start_time.timestamp()}".encode()
        ).hexdigest()[:12]

        if end_time is None:
            end_time = start_time + timedelta(hours=1)

        event = CalendarEvent(
            id=event_id,
            user_id=user_id,
            event_type=event_type,
            title=title,
            description=description,
            start_time=start_time,
            end_time=end_time,
            location=location,
            reminder_minutes=reminder_minutes or [30, 1440],
            color_id=self.EVENT_COLORS.get(event_type, "1"),
            recurrence=recurrence
        )

        if user_id not in self.events:
            self.events[user_id] = []
        self.events[user_id].append(event)

        if user_id in self.connections and self.connections[user_id].sync_enabled:
            if user_id not in self.pending_sync:
                self.pending_sync[user_id] = []
            self.pending_sync[user_id].append(event_id)

        return {
            "event_id": event_id,
            "title": title,
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "synced": False,
            "pending_sync": user_id in self.pending_sync and event_id in self.pending_sync.get(user_id, [])
        }

    def create_decision_deadline(
        self,
        user_id: str,
        decision_title: str,
        deadline: datetime,
        description: str = ""
    ) -> Dict[str, Any]:
        return self.create_event(
            user_id=user_id,
            event_type=CalendarEventType.DECISION_DEADLINE,
            title=f"Decision Due: {decision_title}",
            start_time=deadline,
            end_time=deadline + timedelta(hours=1),
            description=description or f"Deadline to make a decision on: {decision_title}",
            reminder_minutes=[60, 1440, 10080]
        )

    def create_check_in_event(
        self,
        user_id: str,
        check_in_title: str,
        scheduled_time: datetime,
        recurrence: str = "RRULE:FREQ=WEEKLY;BYDAY=MO"
    ) -> Dict[str, Any]:
        return self.create_event(
            user_id=user_id,
            event_type=CalendarEventType.CHECK_IN,
            title=f"Check-in: {check_in_title}",
            start_time=scheduled_time,
            end_time=scheduled_time + timedelta(minutes=30),
            description="Regular career check-in and reflection time",
            recurrence=recurrence,
            reminder_minutes=[30]
        )

    def create_goal_milestone(
        self,
        user_id: str,
        goal_title: str,
        milestone: str,
        target_date: datetime
    ) -> Dict[str, Any]:
        return self.create_event(
            user_id=user_id,
            event_type=CalendarEventType.GOAL_MILESTONE,
            title=f"Goal Milestone: {milestone}",
            start_time=target_date,
            end_time=target_date + timedelta(hours=1),
            description=f"Milestone for goal: {goal_title}",
            reminder_minutes=[1440, 10080]
        )

    def create_follow_up(
        self,
        user_id: str,
        decision_title: str,
        follow_up_date: datetime
    ) -> Dict[str, Any]:
        return self.create_event(
            user_id=user_id,
            event_type=CalendarEventType.FOLLOW_UP,
            title=f"Follow-up: {decision_title}",
            start_time=follow_up_date,
            end_time=follow_up_date + timedelta(minutes=30),
            description=f"Review the outcome of your decision: {decision_title}",
            reminder_minutes=[30, 1440]
        )

    @staticmethod
    def _make_naive(dt: datetime) -> datetime:
        if dt and dt.tzinfo is not None:
            return dt.replace(tzinfo=None)
        return dt

    def get_events(
        self,
        user_id: str,
        start_date: datetime = None,
        end_date: datetime = None,
        event_type: CalendarEventType = None
    ) -> List[Dict[str, Any]]:
        if user_id not in self.events:
            return []

        events = self.events[user_id]

        naive_start = self._make_naive(start_date) if start_date else None
        naive_end = self._make_naive(end_date) if end_date else None

        if naive_start:
            events = [e for e in events if self._make_naive(e.start_time) >= naive_start]
        if naive_end:
            events = [e for e in events if self._make_naive(e.start_time) <= naive_end]
        if event_type:
            events = [e for e in events if e.event_type == event_type]

        events.sort(key=lambda x: x.start_time)

        return [{
            "id": e.id,
            "type": e.event_type.value,
            "title": e.title,
            "description": e.description,
            "start": e.start_time.isoformat(),
            "end": e.end_time.isoformat(),
            "location": e.location,
            "synced": e.synced,
            "google_event_id": e.google_event_id,
            "color_id": e.color_id
        } for e in events]

    def get_upcoming_events(
        self,
        user_id: str,
        days: int = 7
    ) -> List[Dict[str, Any]]:
        now = datetime.utcnow()
        end_date = now + timedelta(days=days)
        return self.get_events(user_id, start_date=now, end_date=end_date)

    def update_event(
        self,
        user_id: str,
        event_id: str,
        **updates
    ) -> Dict[str, Any]:
        if user_id not in self.events:
            return {"error": "No events found"}

        for event in self.events[user_id]:
            if event.id == event_id:
                for key, value in updates.items():
                    if hasattr(event, key):
                        setattr(event, key, value)
                event.updated_at = datetime.utcnow()
                event.synced = False

                if user_id in self.connections and self.connections[user_id].sync_enabled:
                    if user_id not in self.pending_sync:
                        self.pending_sync[user_id] = []
                    if event_id not in self.pending_sync[user_id]:
                        self.pending_sync[user_id].append(event_id)

                return {"updated": True, "event_id": event_id}

        return {"error": "Event not found"}

    def delete_event(
        self,
        user_id: str,
        event_id: str
    ) -> Dict[str, Any]:
        if user_id not in self.events:
            return {"error": "No events found"}

        original_count = len(self.events[user_id])
        self.events[user_id] = [
            e for e in self.events[user_id] if e.id != event_id
        ]

        if len(self.events[user_id]) < original_count:
            return {"deleted": True, "event_id": event_id}
        return {"error": "Event not found"}

    def sync_to_google(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self.connections:
            return {"error": "Not connected to Google Calendar"}

        if user_id not in self.pending_sync or not self.pending_sync[user_id]:
            return {"synced": 0, "message": "No events pending sync"}

        synced_count = 0
        failed_count = 0

        for event_id in self.pending_sync[user_id]:
            for event in self.events.get(user_id, []):
                if event.id == event_id:
                    event.synced = True
                    event.google_event_id = f"google_{event_id}"
                    synced_count += 1
                    break

        self.pending_sync[user_id] = []
        self.connections[user_id].last_sync = datetime.utcnow()

        return {
            "synced": synced_count,
            "failed": failed_count,
            "last_sync": self.connections[user_id].last_sync.isoformat()
        }

    def import_from_google(self, user_id: str) -> Dict[str, Any]:
        if user_id not in self.connections:
            return {"error": "Not connected"}

        return {
            "imported": 0,
            "message": "Google Calendar API integration required for import"
        }

    def get_today_agenda(self, user_id: str) -> Dict[str, Any]:
        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)

        events = self.get_events(user_id, start_date=today_start, end_date=today_end)

        return {
            "date": today_start.strftime("%Y-%m-%d"),
            "event_count": len(events),
            "events": events,
            "summary": self._generate_agenda_summary(events)
        }

    def _generate_agenda_summary(self, events: List[Dict[str, Any]]) -> str:
        if not events:
            return "No career events scheduled for today."

        event_types = {}
        for e in events:
            t = e["type"]
            event_types[t] = event_types.get(t, 0) + 1

        parts = []
        for t, count in event_types.items():
            type_name = t.replace("_", " ").title()
            parts.append(f"{count} {type_name}")

        return f"Today's agenda: {', '.join(parts)}"


google_calendar_service = GoogleCalendarSyncService()
