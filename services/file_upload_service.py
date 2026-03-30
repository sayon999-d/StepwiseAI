import os
import json
import hashlib
import mimetypes
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import asyncio

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import PyPDF2
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False


@dataclass
class UploadedFile:
    id: str
    original_name: str
    file_type: str
    mime_type: str
    size: int
    upload_time: datetime
    user_id: str
    extracted_content: str = ""
    metadata: Dict = field(default_factory=dict)
    processing_status: str = "pending"
    error_message: str = ""


@dataclass
class UserContext:
    user_id: str
    uploaded_files: List[UploadedFile] = field(default_factory=list)
    extracted_knowledge: List[str] = field(default_factory=list)
    training_data: List[Dict] = field(default_factory=list)
    context_summary: str = ""
    last_updated: datetime = field(default_factory=datetime.utcnow)

    def get_context_for_ai(self) -> str:
        if not self.extracted_knowledge and not self.context_summary:
            return ""

        context_parts = []

        if self.context_summary:
            context_parts.append(f"User Context Summary:\n{self.context_summary}")

        if self.extracted_knowledge:
            recent_knowledge = self.extracted_knowledge[-5:]
            context_parts.append("Recent Uploaded Content:\n" + "\n---\n".join(recent_knowledge))

        return "\n\n".join(context_parts)


class FileUploadService:
    MAX_FILE_SIZE = 50 * 1024 * 1024  
    ALLOWED_EXTENSIONS = {
        'text': ['.txt', '.md', '.csv', '.json', '.xml', '.yaml', '.yml'],
        'document': ['.pdf', '.doc', '.docx'],
        'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'],
        'code': ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.html', '.css'],
        'data': ['.xlsx', '.xls', '.parquet'],
        'video': ['.mp4', '.webm', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    }

    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.user_contexts: Dict[str, UserContext] = {}
        self.files_db: Dict[str, UploadedFile] = {}

    def get_user_context(self, user_id: str) -> UserContext:
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = UserContext(user_id=user_id)
        return self.user_contexts[user_id]

    def _get_file_category(self, extension: str) -> str:
        ext = extension.lower()
        for category, extensions in self.ALLOWED_EXTENSIONS.items():
            if ext in extensions:
                return category
        return "unknown"

    def _is_allowed_file(self, filename: str) -> bool:
        ext = Path(filename).suffix.lower()
        all_extensions = []
        for extensions in self.ALLOWED_EXTENSIONS.values():
            all_extensions.extend(extensions)
        return ext in all_extensions or self._get_file_category(ext) != "unknown"

    def _generate_file_id(self, content: bytes, filename: str) -> str:
        hash_content = hashlib.sha256(content[:1024] + filename.encode()).hexdigest()[:16]
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return f"{timestamp}_{hash_content}"

    async def process_file(self, file_content: bytes, filename: str, user_id: str) -> UploadedFile:
        if len(file_content) > self.MAX_FILE_SIZE:
            raise ValueError(f"File size exceeds maximum allowed size of {self.MAX_FILE_SIZE // (1024*1024)}MB")

        file_id = self._generate_file_id(file_content, filename)
        extension = Path(filename).suffix.lower()
        file_type = self._get_file_category(extension)
        mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        uploaded_file = UploadedFile(
            id=file_id,
            original_name=filename,
            file_type=file_type,
            mime_type=mime_type,
            size=len(file_content),
            upload_time=datetime.utcnow(),
            user_id=user_id,
            processing_status="processing"
        )

        file_path = self.upload_dir / f"{file_id}{extension}"
        with open(file_path, "wb") as f:
            f.write(file_content)

        uploaded_file.metadata["saved_path"] = str(file_path)

        try:
            extracted = await self._extract_content(file_content, filename, file_type, mime_type)
            uploaded_file.extracted_content = extracted
            uploaded_file.processing_status = "completed"

            user_context = self.get_user_context(user_id)
            user_context.uploaded_files.append(uploaded_file)

            content_lower = extracted.lower()
            filename_lower = filename.lower() if filename else ""

            is_resume = any(k in content_lower for k in ["resume", "curriculum vitae", "cv", "work experience", "education"]) \
                or any(k in filename_lower for k in ["resume", "cv", "curriculum_vitae", "curriculum-vitae"])

            min_length_for_parse = 100 if not any(k in filename_lower for k in ["resume", "cv"]) else 20

            if is_resume and len(content_lower) > min_length_for_parse:
                try:
                    from services.resume_parser_service import resume_parser_service
                    parsed_data = resume_parser_service.parse_resume(user_id, extracted, filename)
                    uploaded_file.metadata["resume_data"] = parsed_data
                    uploaded_file.metadata["is_resume"] = True
                except Exception as e:
                    print(f"Resume parsing failed (non-critical): {e}")

            if extracted:
                knowledge_entry = f"[From {filename}]:\n{extracted[:2000]}"
                user_context.extracted_knowledge.append(knowledge_entry)
                user_context.last_updated = datetime.utcnow()

                await self._update_context_summary(user_context)

        except Exception as e:
            uploaded_file.processing_status = "failed"
            uploaded_file.error_message = str(e)

        self.files_db[file_id] = uploaded_file

        return uploaded_file

    async def _extract_content(self, content: bytes, filename: str, file_type: str, mime_type: str) -> str:
        if file_type == "text" or file_type == "code":
            return await self._extract_text(content)

        elif file_type == "image":
            return await self._extract_image_info(content, filename)

        elif file_type == "document":
            if filename.endswith(".pdf"):
                return await self._extract_pdf(content)
            else:
                return f"Document uploaded: {filename} (Content extraction limited for this format)"

        elif file_type == "data":
            return await self._extract_data_file(content, filename)

        elif file_type == "video":
            return await self._extract_video_info(content, filename)

        return f"File uploaded: {filename}"

    async def _extract_text(self, content: bytes) -> str:
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            try:
                text = content.decode('latin-1')
            except:
                text = "Unable to decode text content"

        if len(text) > 50000:
            text = text[:25000] + "\n\n[...content truncated...]\n\n" + text[-25000:]

        return text

    async def _extract_image_info(self, content: bytes, filename: str) -> str:
        info = f"Image file: {filename}\n"
        info += f"Size: {len(content)} bytes\n"

        if HAS_PIL:
            try:
                import io
                img = Image.open(io.BytesIO(content))
                info += f"Dimensions: {img.width}x{img.height}\n"
                info += f"Format: {img.format}\n"
                info += f"Mode: {img.mode}\n"

                info += "\n[Image content can be analyzed for career-related context. Please describe what this image represents for your career decision.]"
            except Exception as e:
                info += f"Could not analyze image: {str(e)}"
        else:
            info += "\n[Image uploaded - please describe its relevance to your career decision]"

        return info

    async def _extract_pdf(self, content: bytes) -> str:
        if not HAS_PYPDF:
            return "PDF uploaded. Content extraction requires PyPDF2. Please describe the document content."

        try:
            import io
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            text_parts = []

            for i, page in enumerate(reader.pages[:20]):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(f"[Page {i+1}]\n{page_text}")

            full_text = "\n\n".join(text_parts)

            if len(full_text) > 50000:
                full_text = full_text[:50000] + "\n\n[...content truncated...]"

            return full_text
        except Exception as e:
            return f"PDF extraction failed: {str(e)}. Please describe the document content."

    async def _extract_data_file(self, content: bytes, filename: str) -> str:
        info = f"Data file: {filename}\n"
        info += f"Size: {len(content)} bytes\n"

        if filename.endswith('.json'):
            try:
                data = json.loads(content.decode('utf-8'))
                if isinstance(data, list):
                    info += f"Contains {len(data)} records\n"
                    if data:
                        info += f"Sample fields: {list(data[0].keys())[:10]}\n"
                elif isinstance(data, dict):
                    info += f"Contains keys: {list(data.keys())[:10]}\n"
            except:
                pass

        info += "\n[Data file uploaded - the AI will consider this context in your career discussions]"
        return info

    async def _extract_video_info(self, content: bytes, filename: str) -> str:
        info = f"Video file: {filename}\n"
        info += f"Size: {len(content) / (1024*1024):.2f} MB\n"

        try:
            import cv2
            import io
            
            temp_path = self.upload_dir / f"temp_{filename}"
            with open(temp_path, 'wb') as f:
                f.write(content)
            
            cap = cv2.VideoCapture(str(temp_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            duration_sec = int(frame_count / fps) if fps > 0 else 0
            duration_min = duration_sec // 60
            
            cap.release()
            
            info += f"Duration: {duration_min} min {duration_sec % 60} sec\n"
            info += f"Resolution: {width}x{height}\n"
            info += f"FPS: {fps:.2f}\n"
            
            try:
                temp_path.unlink()
            except:
                pass
                
        except Exception as e:
            info += f"Could not analyze video: {str(e)}\n"
        
        info += "\n[Video file uploaded - the system will extract and analyze content for training. Transcript will be generated if available.]"
        return info

    async def _update_context_summary(self, user_context: UserContext):
        if len(user_context.extracted_knowledge) > 3:
            all_content = "\n---\n".join(user_context.extracted_knowledge[-5:])

            summary = f"User has provided {len(user_context.uploaded_files)} files with content about: "

            topics = []
            content_lower = all_content.lower()
            if any(word in content_lower for word in ['resume', 'cv', 'experience', 'skills']):
                topics.append("resume/CV")
            if any(word in content_lower for word in ['offer', 'salary', 'compensation', 'benefits']):
                topics.append("job offers")
            if any(word in content_lower for word in ['company', 'organization', 'employer']):
                topics.append("companies/employers")
            if any(word in content_lower for word in ['learning', 'course', 'training', 'certificate']):
                topics.append("training/education")

            if topics:
                summary += ", ".join(topics)
            else:
                summary += "career-related content"

            user_context.context_summary = summary

    async def train_system_with_media(self, user_id: str, media_source) -> bool:
        try:
            from services.rag_service import RAGService
            from services.fine_tuning_service import fine_tuning_service
            
            rag_service = RAGService()
            
            if media_source.extracted_content:
                success = rag_service.add_media_content(
                    media_id=media_source.id,
                    media_type=media_source.source_type,
                    title=media_source.title,
                    content=media_source.extracted_content,
                    source_url=media_source.original_url
                )
                
                if success and media_source.transcript:
                    rag_service.add_transcript_chunks(
                        video_id=media_source.id,
                        transcript=media_source.transcript,
                        title=media_source.title
                    )
            
            if media_source.extracted_content and len(media_source.extracted_content) > 100:
                training_content = media_source.extracted_content[:1000]
                training_completion = f"Summary: This {media_source.source_type} contains: {media_source.description[:200]}"
                
                fine_tuning_service.collect_training_pair(
                    user_id=user_id,
                    prompt=f"Process this content: {media_source.title}",
                    completion=training_completion,
                    feedback_score=4
                )
            
            return True
            
        except Exception as e:
            print(f"Error training system with media: {str(e)}")
            return False

    def get_file_info(self, file_id: str) -> Optional[UploadedFile]:
        return self.files_db.get(file_id)

    def get_user_files(self, user_id: str) -> List[UploadedFile]:
        context = self.get_user_context(user_id)
        return context.uploaded_files

    def clear_user_context(self, user_id: str):
        if user_id in self.user_contexts:
            del self.user_contexts[user_id]

    async def process_url_input(self, url: str, user_id: str) -> Dict:
        try:
            from urllib.parse import urlparse
            import ipaddress as _ipa
            import socket

            parsed = urlparse(url)
            hostname = parsed.hostname
            if not hostname:
                return {"success": False, "error": "Invalid URL"}

            try:
                resolved_ips = socket.getaddrinfo(hostname, None)
                for _, _, _, _, sockaddr in resolved_ips:
                    ip = _ipa.ip_address(sockaddr[0])
                    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                        return {
                            "success": False,
                            "error": "URL resolves to a private/internal address and is blocked for security."
                        }
            except (socket.gaierror, ValueError):
                pass  

            if parsed.scheme not in ("http", "https"):
                return {"success": False, "error": "Only HTTP and HTTPS URLs are allowed"}

            from services.media_ingestion_service import media_ingestion_service
            
            media_source = await media_ingestion_service.process_url(url, user_id)
            
            user_context = self.get_user_context(user_id)
            
            if media_source.extracted_content:
                knowledge_entry = f"[From {media_source.source_type}: {media_source.title}]:\n{media_source.extracted_content[:2000]}"
                user_context.extracted_knowledge.append(knowledge_entry)
                user_context.last_updated = datetime.utcnow()
                await self._update_context_summary(user_context)
            
            return {
                "success": True,
                "media_id": media_source.id,
                "title": media_source.title,
                "type": media_source.source_type,
                "status": media_source.processing_status,
                "error": media_source.error_message if media_source.processing_status == "failed" else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def process_video_file_input(self, file_content: bytes, filename: str, user_id: str) -> Dict:
        try:
            from services.media_ingestion_service import media_ingestion_service
            
            media_source = await media_ingestion_service.process_video_file(
                file_content, filename, user_id
            )
            
            user_context = self.get_user_context(user_id)
            
            if media_source.extracted_content:
                knowledge_entry = f"[From video: {media_source.title}]:\n{media_source.extracted_content[:2000]}"
                user_context.extracted_knowledge.append(knowledge_entry)
                user_context.last_updated = datetime.utcnow()
                await self._update_context_summary(user_context)
            
            return {
                "success": True,
                "media_id": media_source.id,
                "title": media_source.title,
                "type": "video",
                "status": media_source.processing_status,
                "error": media_source.error_message if media_source.processing_status == "failed" else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
