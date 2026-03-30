from typing import Dict, List, Optional, Any, BinaryIO
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import base64
import io
import os
import tempfile

class VoiceProvider(str, Enum):
    WHISPER = "whisper"
    BROWSER = "browser"
    FALLBACK = "fallback"

class TTSProvider(str, Enum):
    GTTS = "gtts"
    PYTTSX3 = "pyttsx3"
    BROWSER = "browser"

@dataclass
class TranscriptionResult:
    text: str
    confidence: float
    language: str
    duration_seconds: float
    provider: VoiceProvider
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class SpeechResult:
    audio_data: bytes
    format: str
    duration_seconds: float
    provider: TTSProvider
    timestamp: datetime = field(default_factory=datetime.utcnow)

class VoiceSpeechService:
    VOICE_PERSONAS = {
        "future_self_calm": {
            "rate": 150,
            "pitch": 1.0,
            "voice": "en-US-Wavenet-D",
            "description": "Calm and reflective future self"
        },
        "future_self_energetic": {
            "rate": 170,
            "pitch": 1.1,
            "voice": "en-US-Wavenet-A",
            "description": "Energetic and enthusiastic future self"
        },
        "future_self_wise": {
            "rate": 130,
            "pitch": 0.9,
            "voice": "en-US-Wavenet-B",
            "description": "Wise and contemplative future self"
        },
        "coach": {
            "rate": 160,
            "pitch": 1.0,
            "voice": "en-US-Wavenet-C",
            "description": "Professional coaching voice"
        }
    }

    def __init__(self):
        self.whisper_model = None
        self.tts_engine = None
        self._initialize_engines()

    def _initialize_engines(self):
        try:
            import whisper
            if not hasattr(whisper, 'load_model'):
                raise ImportError("Installed 'whisper' package is not OpenAI Whisper (missing load_model)")
            self.whisper_model = whisper.load_model("base")
            self.stt_provider = VoiceProvider.WHISPER
            print("Whisper STT initialized")
        except ImportError:
            self.stt_provider = VoiceProvider.BROWSER
        except Exception:
            self.stt_provider = VoiceProvider.BROWSER

        try:
            import pyttsx3
            self.tts_engine = pyttsx3.init()
            self.tts_provider = TTSProvider.PYTTSX3
            print("pyttsx3 TTS initialized")
        except Exception:
            try:
                from gtts import gTTS
                self.tts_provider = TTSProvider.GTTS
                print("gTTS TTS initialized")
            except ImportError:
                self.tts_provider = TTSProvider.BROWSER
                print("Using browser-based TTS")

    async def transcribe_audio(
        self,
        audio_data: bytes,
        audio_format: str = "webm",
        language: str = "en"
    ) -> TranscriptionResult:
        if self.whisper_model:
            return await self._transcribe_with_whisper(audio_data, audio_format, language)
        else:
            return TranscriptionResult(
                text="",
                confidence=0,
                language=language,
                duration_seconds=0,
                provider=VoiceProvider.BROWSER
            )

    async def _transcribe_with_whisper(
        self,
        audio_data: bytes,
        audio_format: str,
        language: str
    ) -> TranscriptionResult:
        with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as tmp_file:
            tmp_file.write(audio_data)
            tmp_path = tmp_file.name

        try:
            result = self.whisper_model.transcribe(tmp_path, language=language)

            segments = result.get("segments", [])
            duration = segments[-1]["end"] if segments else 0

            return TranscriptionResult(
                text=result["text"].strip(),
                confidence=0.9,
                language=result.get("language", language),
                duration_seconds=duration,
                provider=VoiceProvider.WHISPER
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def text_to_speech(
        self,
        text: str,
        persona: str = "future_self_calm",
        output_format: str = "mp3"
    ) -> SpeechResult:
        persona_config = self.VOICE_PERSONAS.get(persona, self.VOICE_PERSONAS["future_self_calm"])

        if self.tts_provider == TTSProvider.PYTTSX3:
            return await self._tts_pyttsx3(text, persona_config, output_format)
        elif self.tts_provider == TTSProvider.GTTS:
            return await self._tts_gtts(text, output_format)
        else:
            return SpeechResult(
                audio_data=b"",
                format=output_format,
                duration_seconds=0,
                provider=TTSProvider.BROWSER
            )

    async def _tts_pyttsx3(self, text: str, config: Dict, output_format: str) -> SpeechResult:
        with tempfile.NamedTemporaryFile(suffix=f".{output_format}", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            self.tts_engine.setProperty('rate', config.get('rate', 150))
            self.tts_engine.save_to_file(text, tmp_path)
            self.tts_engine.runAndWait()

            with open(tmp_path, 'rb') as f:
                audio_data = f.read()

            words = len(text.split())
            duration = words / (config.get('rate', 150) / 60)

            return SpeechResult(
                audio_data=audio_data,
                format=output_format,
                duration_seconds=duration,
                provider=TTSProvider.PYTTSX3
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    async def _tts_gtts(self, text: str, output_format: str) -> SpeechResult:
        from gtts import gTTS

        tts = gTTS(text=text, lang='en')

        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_data = audio_buffer.getvalue()

        words = len(text.split())
        duration = words / 2.5

        return SpeechResult(
            audio_data=audio_data,
            format="mp3",
            duration_seconds=duration,
            provider=TTSProvider.GTTS
        )

    def get_available_personas(self) -> List[Dict]:
        return [
            {"id": key, **value}
            for key, value in self.VOICE_PERSONAS.items()
        ]

    def get_service_status(self) -> Dict[str, Any]:
        return {
            "stt_provider": self.stt_provider.value,
            "tts_provider": self.tts_provider.value,
            "whisper_available": self.whisper_model is not None,
            "personas_available": list(self.VOICE_PERSONAS.keys())
        }

class VoiceJournalService:
    def __init__(self, voice_service: VoiceSpeechService):
        self.voice_service = voice_service
        self.voice_entries: Dict[str, List[Dict]] = {}

    async def process_voice_entry(
        self,
        user_id: str,
        audio_data: bytes,
        audio_format: str = "webm"
    ) -> Dict[str, Any]:
        transcription = await self.voice_service.transcribe_audio(audio_data, audio_format)

        entry = {
            "id": f"voice_{user_id}_{datetime.utcnow().timestamp()}",
            "user_id": user_id,
            "transcription": transcription.text,
            "confidence": transcription.confidence,
            "duration": transcription.duration_seconds,
            "recorded_at": datetime.utcnow().isoformat()
        }

        if user_id not in self.voice_entries:
            self.voice_entries[user_id] = []
        self.voice_entries[user_id].append(entry)

        return entry

    def get_voice_entries(self, user_id: str) -> List[Dict]:
        return self.voice_entries.get(user_id, [])

class FutureSelfVoiceService:
    def __init__(self, voice_service: VoiceSpeechService):
        self.voice_service = voice_service
        self.session_personas: Dict[str, str] = {}

    def set_session_persona(self, session_id: str, persona: str):
        if persona in self.voice_service.VOICE_PERSONAS:
            self.session_personas[session_id] = persona

    def get_session_persona(self, session_id: str) -> str:
        return self.session_personas.get(session_id, "future_self_calm")

    async def voice_to_text(self, audio_data: bytes, audio_format: str = "webm") -> str:
        result = await self.voice_service.transcribe_audio(audio_data, audio_format)
        return result.text

    async def text_to_voice(self, text: str, session_id: str) -> bytes:
        persona = self.get_session_persona(session_id)
        result = await self.voice_service.text_to_speech(text, persona)
        return result.audio_data

    def select_persona_for_scenario(self, scenario: str, emotional_state: str) -> str:
        if scenario == "optimistic" and emotional_state in ["content", "excited", "hopeful"]:
            return "future_self_energetic"
        elif scenario == "pessimistic" or emotional_state in ["reflective", "cautious"]:
            return "future_self_wise"
        else:
            return "future_self_calm"

voice_speech_service = VoiceSpeechService()
voice_journal_service = VoiceJournalService(voice_speech_service)
future_self_voice_service = FutureSelfVoiceService(voice_speech_service)
