import json
from typing import Optional
from pydantic import BaseModel


class AudioProjectCreate(BaseModel):
    user_id: int
    name: str
    original_audio_path: Optional[str] = "undefined"
    generated_audio_path: Optional[str] = "undefined"
    duration: Optional[float] = 0.
    voice: Optional[str] = "hn_female_ngochuyen_full_48k-fhg"


class AudioProjectUpdate(BaseModel):
    id: int
    user_id: Optional[int] = None
    name: Optional[str] = None
    original_audio_path: Optional[str] = None
    generated_audio_path: Optional[str] = None
    duration: Optional[float] = None
    voice: Optional[str] = None
    status: Optional[str] = None
    percentage: Optional[int] = None
    processing_length: Optional[float] = None


class AudioProjectResponse(BaseModel):
    id: int


class TranscribeResponse(BaseModel):
    success: bool
    message: str


class AudioSegmentUpdate(BaseModel):
    id: int
    project_id: Optional[int] = None
    start: Optional[float] = None
    end: Optional[float] = None
    text: Optional[str] = None
    original_audio_path: Optional[str] = None
    generated_audio_path: Optional[str] = None
    has_modified: Optional[bool] = True


class AudioSegmentUpdateResponse(BaseModel):
    id: int


class AudioSegmentResponse(BaseModel):
    id: int
    project_id: int
    start: float
    end: float
    text: str
    original_audio_path: str
    generated_audio_path: str


class SplitAudioSchema(BaseModel):
    id: int
    split_point: float


class MergeAudioSchema(BaseModel):
    id_1: int
    id_2: int
