import os


class Config:
    # graphql_endpoint = "http://localhost:8005/api/rest"
    graphql_endpoint = os.environ.get("GRAPHQL_ENDPOINT")
    punc_model_path = "./weights/checkpoint_30501"
    data_dir = "data"
    # vbee_audio_path = "/vtca/database_wav2lip/temp"
    vbee_audio_path = "/app/database_wav2lip/temp"
    # vbee_api_endpoint = "http://localhost:9998"
    vbee_api_endpoint = os.environ.get("VBEE_API_ENDPOINT")
    original_audio_dir = "original_audio"
    generated_audio_dir ="generated_audio"
    trim_generated_audio_dir = "trim_generated_audio"
    # sample_rate = 22050
    sample_rate = int(os.environ.get("SAMPLE_RATE"))
    inaudiable_text = "(không thể nhận diện)"

    project_idle_key = "IDLE"
    project_segment_key = "SEGMENT"
    project_export_key = "EXPORT"
