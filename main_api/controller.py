import os
from fastapi import UploadFile, File, Body, APIRouter, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from schemas import AudioProjectCreate, AudioProjectUpdate, AudioProjectResponse, TranscribeResponse, AudioSegmentUpdate, AudioSegmentResponse, AudioSegmentUpdateResponse, SplitAudioSchema, MergeAudioSchema
from services import AudioProjectService, AudioSegmentService
from config import Config


audio_align_router = APIRouter()
audio_segment_service = AudioSegmentService()
audio_project_service = AudioProjectService(audio_segment_service)

AUDIO_FILE_TAG = "Audio File"
AUDIO_PROJECT_TAG = "Audio Project"
AUDIO_SEGMENT_TAG = "Audio Segment"


@audio_align_router.get(
    "/audio",
    summary="Get audio from path",
    tags=[AUDIO_FILE_TAG]
)
async def get_audio(path: str):
    return FileResponse(path)

@audio_align_router.get(
    "/generated-audio",
    summary="Get generated audio from audio API",
    tags=[AUDIO_FILE_TAG]
)
async def get_gen_audio(audio_id: str):
    return FileResponse(audio_segment_service.get_generated_audio_path(audio_id))


@audio_align_router.post(
    "/audio-project",
    summary="Create audio project",
    tags=[AUDIO_PROJECT_TAG]
)
async def create_audio_project(item: AudioProjectCreate):
    return audio_project_service.create_audio_project(item)


@audio_align_router.put(
    "/audio-project",
    summary="Update audio project",
    tags=[AUDIO_PROJECT_TAG]
)
async def update_audio_project(item: AudioProjectUpdate):
    return audio_project_service.update_audio_project(item)


@audio_align_router.delete(
    "/audio-project",
    summary="Delete audio project",
    tags=[AUDIO_PROJECT_TAG]
)
async def delete_audio_project(id: int):
    return audio_project_service.delete_audio_project(id)


@audio_align_router.post(
    "/generate-project-audio",
    summary="Generate final audio for the project",
    tags=[AUDIO_PROJECT_TAG]
)
async def generate_project_audio(id: int):
    return audio_project_service.generate_final_audio(id)

@audio_align_router.post(
    "/export-srt",
    summary="Create SRT file",
    tags=[AUDIO_PROJECT_TAG]
)
async def export_srt(id: int):
    return audio_project_service.export_srt(id)

@audio_align_router.get(
    "/audio-project-pagination",
    summary="list of audio projects for pagination",
    tags=[AUDIO_PROJECT_TAG]
)
async def get_audio_projects(user_id: int, start: int, end: int):
    return audio_project_service.get_audio_projects(user_id, start, end)


@audio_align_router.get(
    "/audio-project-count",
    summary="Count number of audio projects for a given user",
    tags=[AUDIO_PROJECT_TAG]
)
async def get_project_count(user_id: int):
    return audio_project_service.get_project_count(user_id)


@audio_align_router.post(
    "/transcribe",
    summary="Segment and transcribe audio",
    tags=[AUDIO_SEGMENT_TAG]
)
def transcribe(
    file: UploadFile = File(...),
    user_id: int = -1,
    project_id: int = -1,
    min_silence_len: int = 500
):
    return audio_segment_service.transcribe(file, user_id, project_id, min_silence_len)

@audio_align_router.post(
    "/re-transcribe",
    summary="Re-segment and transcribe audio",
    tags=[AUDIO_SEGMENT_TAG]
)
def re_transcribe(
    user_id: int = -1,
    project_id: int = -1,
    min_silence_len: int = 500
):
    return audio_segment_service.re_transcribe(user_id, project_id, min_silence_len)

@audio_align_router.get(
    "/audio-segment",
    summary="Get audio segment",
    tags=[AUDIO_SEGMENT_TAG]
)
async def get_audio_segment(id: int):
    return audio_segment_service.get_audio_segment(id)


@audio_align_router.put(
    "/audio-segment",
    summary="Update audio segment",
    tags=[AUDIO_SEGMENT_TAG]
)
async def update_audio_segment(item: AudioSegmentUpdate):
    return audio_segment_service.update_audio_segment(item)


@audio_align_router.post(
    "/split-audio-segment",
    summary="Split audio segment into 2 smaller chunks",
    tags=[AUDIO_SEGMENT_TAG]
)
async def split_audio_segment(item: SplitAudioSchema):
    return audio_segment_service.split_audio(item.id, item.split_point)

@audio_align_router.post(
    "/merge-audio-segment",
    summary="Merge audio segments into 1 chunk",
    tags=[AUDIO_SEGMENT_TAG]
)
async def merge_audio_segments(item: MergeAudioSchema):
    return audio_segment_service.merge_audios(item.id_1, item.id_2)

@audio_align_router.post(
    "/generate-segment-audio",
    summary="Generate AI-spoken audio",
    tags=[AUDIO_SEGMENT_TAG]
)
async def generate_segment_audio(id: int):
    return audio_segment_service.generate_audio(id)


@audio_align_router.get(
    "/audio-segment-pagination",
    summary="list of audio segments for pagination",
    tags=[AUDIO_SEGMENT_TAG]
)
async def get_audio_segments(project_id: int, start: int, end: int):
    return audio_segment_service.get_audio_segments(project_id, start, end)


@audio_align_router.get(
    "/audio-count",
    summary="Count number of audio segments for a given project",
    tags=[AUDIO_SEGMENT_TAG]
)
async def get_audio_count(project_id: int):
    return audio_segment_service.get_audio_count(project_id)
