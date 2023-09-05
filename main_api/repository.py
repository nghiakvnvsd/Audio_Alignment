import os
import requests
import json
from schemas import AudioProjectCreate, AudioProjectUpdate, AudioSegmentUpdate
from config import Config


class AudioProjectRepo:
    def __init__(self):
        self.graphql_endpoint = Config.graphql_endpoint

    def get_audio_project(self, id: int):
        res = requests.get(f"{self.graphql_endpoint}/get-audio-project?id={id}")
        return res.json()["audio_project"][0]

    def create_audio_project(self, item: AudioProjectCreate):
        res = requests.post(f"{self.graphql_endpoint}/create-audio-project", json.dumps(item.dict()))
        return {
            "id": res.json()["insert_audio_project_one"]["id"]
        }

    def update_audio_project(self, item: AudioProjectUpdate):
        data = {
            "id": item.id,
            "changes": dict((k, v) for k, v in item.dict().items() if k != "id" and v is not None)
        }
        res = requests.put(f"{self.graphql_endpoint}/update-audio-project", json.dumps(data))

        return {
            "id": res.json()["update_audio_project"]["returning"][0]["id"]
        }

    def delete_audio_project(self, id: int):
        res = requests.delete(f"{self.graphql_endpoint}/delete-audio-project", data=json.dumps({ "id": id }))
        return {
            "id": res.json()["delete_audio_project"]["returning"][0]["id"]
        }

    def get_audio_projects(self, user_id: int, start: int, end: int):
        '''
        user_id: id of the target user
        start: start index of the audio project in the list
        end: end index of the audio project in the list

        Return
        A list of requested audio projects (JSON) in the given range for this user
        '''
        raise NotImplementedError()
        return []

    def get_project_count(self, user_id: int):
        '''
        user_id: id of the target user

        Return
        The number of audio projects of this user
        '''
        raise NotImplementedError()
        return 0


class AudioSegmentRepo:

    def __init__(self):
        self.graphql_endpoint = Config.graphql_endpoint
        self.vbee_audio_path = Config.vbee_audio_path

    def create_audio_segment(self, project_id, start, end, text, chunk_path):
        res = requests.post(f"{self.graphql_endpoint}/create-audio-segment", {
            "project_id": project_id,
            "start": start,
            "end": end,
            "text": text,
            "original_audio_path": chunk_path,
            "generated_audio_path": "undefined"
        })

        return res.json()["insert_audio_segment_one"]

    def update_audio_segment(self, item: AudioSegmentUpdate):
        data = {
            "id": item.id,
            "changes": dict((k, v) for k, v in item.dict().items() if k != "id" and v is not None)
        }
        res = requests.put(f"{self.graphql_endpoint}/update-audio-segment", json.dumps(data))

        return {
            "id": res.json()["update_audio_segment"]["returning"][0]["id"]
        }

    def delete_audio_segment(self, id: int):
        res = requests.delete(f"{self.graphql_endpoint}/delete-audio-segment", data=json.dumps({ "id": id }))
        return {
            "id": res.json()["delete_audio_segment"]["returning"][0]["id"]
        }

    def get_audio_segment(self, id: int):
        print("id", id)
        print(f"{self.graphql_endpoint}/get-audio-segment?id={id}")
        res = requests.get(f"{self.graphql_endpoint}/get-audio-segment?id={id}")
        print(res.json())
        return res.json()["audio_segment"][0]

    def get_all_audio_segments(self, id: int):
        '''
        id: id of the target project

        Return
        A list of all audio segments (JSON) for this project
        '''
        res = requests.get(f"{self.graphql_endpoint}/audio-segments/{id}")
        return res.json()["audio_segment"]

    def get_audio_segments(self, project_id: int, start: int, end: int):
        '''
        project_id: id of the target project
        start: start index of the audio segment in the list
        end: end index of the audio segment in the list

        Return
        A list of requested audio segments (JSON) in the given range for this project
        '''
        raise NotImplementedError()
        return []

    def get_audio_count(self, project_id: int):
        '''
        project_id: id of the target project

        Return
        The number of audio segments in this project
        '''
        raise NotImplementedError()
        return 0

    def get_generated_audio_path(self, audio_id: str):
        return os.path.join(self.vbee_audio_path, audio_id + ".wav")
