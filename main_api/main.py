from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from controller import audio_align_router

app = FastAPI()
app.include_router(audio_align_router)

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

