import asyncio
import shutil
from time import time
import librosa
import soundfile as sf
import numpy as np
import subprocess
import json
import requests
from loguru import logger
from tqdm import tqdm
from pydub import AudioSegment, silence
import os
import io
from fastapi import HTTPException, status
from fastapi.responses import FileResponse, Response
import speech_recognition
from punctuation_predictor import PunctuationPredictor
from schemas import AudioProjectCreate, AudioProjectUpdate, AudioProjectResponse, AudioSegmentUpdate
from config import Config
from repository import AudioProjectRepo, AudioSegmentRepo


def get_audio_length(audio_path): # in seconds
    try:
        result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', audio_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        return float(result.stdout)
    except:
        pass
    return 0.01


def remove_folder_content(folder):
    for filename in os.listdir(folder):
        if "full_audio" in filename:
            continue
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def format_time(millis):
    millis = int(millis)
    seconds=(millis/1000)%60
    seconds = int(seconds)
    minutes=(millis/(1000*60))%60
    minutes = int(minutes)
    hours=(millis/(1000*60*60))%24

    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{millis % 1000:03}"

class AudioProjectService:

    def __init__(self, audio_segment_service):
        self.data_dir = Config.data_dir
        self.original_audio_dir = Config.original_audio_dir
        self.generated_audio_dir = Config.generated_audio_dir
        self.trim_generated_audio_dir = Config.trim_generated_audio_dir
        self.repo = AudioProjectRepo()
        self.segment_repo = AudioSegmentRepo()
        self.audio_segment_service = audio_segment_service
        self.sample_rate = Config.sample_rate

    def create_audio_project(self, item: AudioProjectCreate):
        try:
            res_data = self.repo.create_audio_project(item)

            # create folder for new project
            project_dir = os.path.join(self.data_dir, f"{item.user_id}", f"{res_data['id']}")
            if not os.path.isdir(project_dir):
                os.makedirs(project_dir)

            original_audio_dir = os.path.join(project_dir, self.original_audio_dir)
            generated_audio_dir = os.path.join(project_dir, self.generated_audio_dir)
            trim_generated_audio_dir = os.path.join(project_dir, self.trim_generated_audio_dir)

            if not os.path.isdir(original_audio_dir): os.makedirs(original_audio_dir)
            if not os.path.isdir(generated_audio_dir): os.makedirs(generated_audio_dir)
            if not os.path.isdir(trim_generated_audio_dir): os.makedirs(trim_generated_audio_dir)

            return {
                "status_code": 200,
                "id": res_data["id"]
            }
        except Exception as e:
            print("============ Create project exception =========\n", e)
            pass

        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                             detail="Cannot create your project at the moment. Please try again in a few minutes.")

    def update_audio_project(self, item: AudioProjectUpdate):
        try:
            if item.voice is not None:  # reset all previous generated audio segments
                audio_list = self.segment_repo.get_all_audio_segments(item.id)
                for audio_data in audio_list:
                    body = {
                        "id": audio_data["id"],
                        "has_modified": True
                    }
                    self.audio_segment_service.update_audio_segment(AudioSegmentUpdate(**body))

            res = self.repo.update_audio_project(item)
            return {
                "status_code": 200,
                "id": res["id"]
            }
        except Exception as e:
            print("============ Update project exception =========\n", e)
            return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                 detail="Cannot update your project at the moment. Please try again in a few minutes.")

        return {
            "status_code": 500,
            "id": -1,
        }

    def delete_audio_project(self, id: int):
        try:
            audio_list = self.segment_repo.get_all_audio_segments(id)
            for audio_data in audio_list:
                self.segment_repo.delete_audio_segment(audio_data["id"])

            # delete project directory
            project_data = self.repo.get_audio_project(id)
            project_dir = os.path.join(self.data_dir, f"{project_data['user_id']}", f"{project_data['id']}")
            shutil.rmtree(project_dir)

            res =  self.repo.delete_audio_project(id)
            return {
                "status_code": 200,
                "id": res["id"]
            }
        except Exception as e:
            print("============ Delete project exception =========\n", e)
            pass

        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                             detail="Cannot delete your project at the moment. Please try again in a few minutes.")

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

    def generate_final_audio(self, project_id):
        '''
        merge all the audio segments of this project based on their time
        '''
        self.repo.update_audio_project(AudioProjectUpdate(**{
            "id": project_id,
            "status": Config.project_export_key,
            "percentage": 0,
            "processing_length": 0
        }))

        st = time()
        # get all the audio segments info for this project
        audio_list = self.segment_repo.get_all_audio_segments(project_id)
        if len(audio_list) == 0:
            self.repo.update_audio_project(AudioProjectUpdate(**{
                "id": project_id,
                "status": Config.project_idle_key,
                "percentage": 100,
                "processing_length": 0
            }))
            return HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                 detail="Không có dữ liệu để xuất bản")

        project_data = self.repo.get_audio_project(project_id)

        # create AI audios
        audio_list.sort(key=lambda x: x["start"])
        regenerate = project_data["generated_audio_path"] == "undefined"
        prev_percentage = 0
        total_audio_length = float(project_data["duration"]) * 1000
        for audio_data in audio_list:
            if audio_data["has_modified"]:
                regenerate = True
            res = self.audio_segment_service.generate_audio(audio_data["id"],
                                                            prev_percentage=prev_percentage,
                                                            total_audio_length=total_audio_length,
                                                            project_id=project_id)
            if res["status_code"] != 200:
                return res

            prev_percentage = res["percentage"]

        if not regenerate: # no need to generate again
            self.repo.update_audio_project(AudioProjectUpdate(**{
                "id": project_id,
                "status": Config.project_idle_key,
                "percentage": 100,
                "processing_length": 0
            }))
            return {
                "status_code": 200,
                "generated_audio_path": project_data["generated_audio_path"]
            }

        self.repo.update_audio_project(AudioProjectUpdate(**{
            "id": project_id,
            "percentage": prev_percentage,
            "processing_length": int(total_audio_length / 99 * 100 * 0.01)
        }))

        gen_time = time() - st
        st = time()
        print(f"========= Merge audios for project - {project_id} ===========")
        # refetch the data
        audio_list = self.segment_repo.get_all_audio_segments(project_id)

        # find the final audio length based on start and end of the first at last audios
        end_list = [item["end"] for item in audio_list]
        final_end = int((max(end_list) + 5) / 1000 * self.sample_rate)

        # create an empty array np.zeros(length of the final audio)
        blank = np.zeros(final_end)

        # load each audio file and paste it to the correct position
        for audio_data in audio_list:
            audio_path = audio_data["generated_audio_path"]
            start_pt = int(audio_data["start"] * self.sample_rate / 1000)
            try:
                au, sr = librosa.load(audio_path, sr=self.sample_rate)
                blank[start_pt:start_pt + len(au)] = au
            except Exception as e:
                print("===== Merge audio exception =====")
                print(e)

        # export the audio file
        project_dir = os.path.join(self.data_dir, f"{project_data['user_id']}", f"{project_data['id']}")
        generated_audio_path = os.path.join(project_dir, self.generated_audio_dir, "full_audio.wav")
        sf.write(generated_audio_path, blank, self.sample_rate)

        # update the generated_audio_path for this project in the database
        self.repo.update_audio_project(AudioProjectUpdate(**{
            "id": project_id,
            "generated_audio_path": generated_audio_path,
            "status": Config.project_idle_key,
            "percentage": 100,
            "processing_length": 0
        }))

        print(f"Generating time {gen_time}")
        print(f"Merging time {time() - st}")

        return {
            "status_code": 200,
            "generated_audio_path": generated_audio_path
        }

    def export_srt(self, project_id):
        '''
        merge all the audio segments of this project based on their time
        '''
        print("====== Creating SRT - project {project_id} ===========")
        # get all the audio segments info for this project
        audio_list = self.segment_repo.get_all_audio_segments(project_id)
        if len(audio_list) == 0:
            return HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                 detail="Không có dữ liệu để xuất bản")

        # create AI audios
        audio_list.sort(key=lambda x: x["start"])

        project_data = self.repo.get_audio_project(project_id)
        project_dir = os.path.join(self.data_dir, f"{project_data['user_id']}", f"{project_data['id']}")

        srt_path = os.path.join(project_dir, "script.srt")
        srt_content = ""
        for i, audio_data in enumerate(audio_list):
            start = int(audio_data["start"])
            end = int(audio_data["end"])
            text = audio_data["text"]

            start_str = format_time(start)
            end_str = format_time(end)

            srt_content += f"{i+1}\n"
            srt_content += f"{start_str} --> {end_str}\n"
            srt_content += f"{text}\n\n"

        with open(srt_path, "w") as f:
            f.write(srt_content)

        print("Done SRT")
        return {
            "status_code": 200,
            "srt_path": srt_path
        }


class AudioSegmentService:

    def __init__(self):
        self.recognizer = speech_recognition.Recognizer()
        self.punc_model = PunctuationPredictor(Config.punc_model_path)
        self.data_dir = Config.data_dir
        self.vbee_api_endpoint = Config.vbee_api_endpoint
        self.vbee_audio_path = Config.vbee_audio_path
        self.original_audio_dir = Config.original_audio_dir
        self.generated_audio_dir = Config.generated_audio_dir
        self.trim_generated_audio_dir = Config.trim_generated_audio_dir
        self.repo = AudioSegmentRepo()
        self.project_repo = AudioProjectRepo()
        self.inaudiable_text = Config.inaudiable_text

    def get_audio_segment(self, id: int):
        try:
            return self.repo.get_audio_segment(id)
        except Exception as e:
            print("============ Get audio segment exception =========\n", e)
            return HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                 detail=f"Audio segment with id {id} does not exist.")

        return None

    def update_audio_segment(self, item: AudioSegmentUpdate):
        try:
            audio_data = self.repo.get_audio_segment(item.id)
            if item.text is not None and item.text == audio_data["text"] and not audio_data["has_modified"]:
                item.has_modified = False

            result = self.repo.update_audio_segment(item)
            return {
                "status_code": 200,
                "id": result["id"]
            }
        except Exception as e:
            print("============ Update Audio Segment exception =========\n", e)
            return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                 detail="Không thể cập nhật thông tin audio. Vui lòng thử lại sau.")

        return {
            "status_code": 500,
            "detail": "Unexpected Error"
        }

    def transcribe_one(self, audio_path):
        with speech_recognition.AudioFile(audio_path) as source:
            audio_data = self.recognizer.record(source)
        transcript = self.recognizer.recognize_google(audio_data, language="vi-VN")
        # add punctuation
        return self.punc_model.predict(transcript)

    async def _transcribe(self, file, outdir, project_id, min_silence_len):
        audio_path = os.path.join(outdir, "full_audio.wav")
        if not os.path.isdir(outdir):
            return {
                "status_code": 400,
                "detail": "Project does not exist"
            }

        time_dict = {}
        self.project_repo.update_audio_project(AudioProjectUpdate(**{
            "id": project_id,
            "status": Config.project_segment_key,
            "percentage": 0,
            "processing_length": 1000
        }))
        st = time()
        duration = 0
        if file is not None:
            request_object_content = await file.read()
            if os.path.isfile(audio_path): os.remove(audio_path)

            byteio = io.BytesIO(request_object_content)
            with open(audio_path, mode='bx') as f:
                f.write(request_object_content)

            duration = get_audio_length(audio_path)
            item = {
                "id": project_id,
                "duration": duration,
                "original_audio_path": audio_path,
            }
            res = self.project_repo.update_audio_project(AudioProjectUpdate(**item))

        time_dict["load"] = f"{duration} {time() - st}"
        st = time()

        self.project_repo.update_audio_project(AudioProjectUpdate(**{
            "id": project_id,
            "percentage": 5,
            "processing_length": 1000
        }))

        try:
            ### 10% ###
            audio = AudioSegment.from_wav(audio_path)
            ranges = silence.detect_silence(audio, min_silence_len=min_silence_len, silence_thresh=audio.dBFS-16)

            time_dict["segment"] = f"{duration} {time() - st}"
            st = time()

            idx = 0
            if len(ranges) > 0 and abs(ranges[-1][1] - len(audio)) > 20:
                ranges.append([len(audio)-1, len(audio)])
            cnt = 0
            for x, y in tqdm(ranges):
                y = max(y-min_silence_len//4, 0)
                chunk = audio[idx:y]
                chunk_id = f"f_{idx}_{y}"
                chunk_path = os.path.join(outdir, f"{chunk_id}.wav")
                chunk.export(chunk_path, format="wav")

                self.project_repo.update_audio_project(AudioProjectUpdate(**{
                    "id": project_id,
                    "percentage": 10 + int(idx/len(audio) * 90),
                    "processing_length": abs(y - idx) # miliseconds
                }))

                cnt += 1
                try:
                    transcript = self.transcribe_one(chunk_path)
                except Exception as e:
                    # no text found, skip this one
                    print("Google error", e)
                    if cnt < len(ranges):
                        continue
                    transcript = self.inaudiable_text

                time_dict[f"transcribe_{cnt}"] = f"{y-idx} {time() - st}"
                st = time()

                logger.success(f"transcript:\n {chunk_path}\n {transcript}")

                # update database
                res = self.repo.create_audio_segment(project_id, idx, y, transcript, chunk_path)
                print(res)
                idx = y
        except Exception as e:
            print("======= Transcribing Error =======\n", e)

        print(time_dict)
        self.project_repo.update_audio_project(AudioProjectUpdate(**{
            "id": project_id,
            "status": Config.project_idle_key,
            "percentage": 100,
            "processing_length": 0
        }))


    def transcribe(self, file, user_id = -1, project_id = -1, min_silence_len = 500):
        print(f"\n======= Transcribing - {project_id} =======\n")
        if user_id == -1 or project_id == -1:
            return HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                 detail="Project does not exist")

        project_dir = os.path.join(self.data_dir, f"{user_id}", f"{project_id}")
        outdir = os.path.join(project_dir, self.original_audio_dir)

        if not os.path.isdir(outdir):
            return HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                 detail="Project does not exist")

        asyncio.run(self._transcribe(file, outdir, project_id, min_silence_len))
        return {
            "status_code": 200,
            "detail": "Processing data"
        }

    def re_transcribe(self, user_id = -1, project_id = -1, min_silence_len = 500):
        print(f"\n======= Re- Transcribing - {project_id} =======\n")
        if user_id == -1 or project_id == -1:
            return HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                 detail="Project does not exist")

        project_dir = os.path.join(self.data_dir, f"{user_id}", f"{project_id}")
        original_audio_dir = os.path.join(project_dir, self.original_audio_dir)
        generated_audio_dir = os.path.join(project_dir, self.generated_audio_dir)
        trim_generated_audio_dir = os.path.join(project_dir, self.trim_generated_audio_dir)

        if not os.path.isdir(original_audio_dir):
            return HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                                 detail="Project does not exist")

        audio_list = self.repo.get_all_audio_segments(project_id)
        for audio_data in audio_list:
            self.repo.delete_audio_segment(audio_data["id"])

        remove_folder_content(original_audio_dir)
        remove_folder_content(generated_audio_dir)
        remove_folder_content(trim_generated_audio_dir)

        asyncio.run(self._transcribe(None, original_audio_dir, project_id, min_silence_len))
        return {
            "status_code": 200,
            "detail": "Processing data"
        }

    def get_time_range_from_filename(self, file_path):
        start, end = os.path.basename(file_path).split('.')[0].replace("f_", "").split("_")
        return int(start), int(end)

    def split_audio(self, audio_segment_id: int, split_point: float):
        '''
        audio_segment_id: id of the audio segment in the database
        split_point: miliseconds counting from the start of the current file
        '''

        # get the original audio info
        original_data = self.repo.get_audio_segment(audio_segment_id)
        start = original_data["start"]
        end = original_data["end"]
        file_path = original_data["original_audio_path"]
        if split_point > original_data["end"]:
            return HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                 detail="Split point must be less than audio length (in miliseconds)")

        # convert to seconds
        split_second = split_point / 1000.
        root_dir = file_path.replace(os.path.basename(file_path), "")
        absolute_split_point = int(original_data["start"] + split_point)

        filename1 = f"f_{int(start)}_{absolute_split_point}.wav"
        file_path1 = os.path.join(root_dir, filename1)
        os.system(f"ffmpeg -y -ss 0 -t {split_second:.10f} -i {file_path} {file_path1}")

        filename2 = f"f_{absolute_split_point}_{int(end)}.wav"
        file_path2 = os.path.join(root_dir, filename2)
        os.system(f"ffmpeg -y -ss {split_second:.10f} -i {file_path} {file_path2}")

        # transcribe the audios
        try:
            text1 = self.transcribe_one(file_path1)
        except:
            text1 = self.inaudiable_text

        try:
            text2 = self.transcribe_one(file_path2)
        except:
            text2 = self.inaudiable_text

        # update the original file to file1
        data1 = {
            "id": audio_segment_id,
            "end": absolute_split_point,
            "text": text1,
            "original_audio_path": file_path1,
            "generated_audio_path": "undefined"
        }
        self.repo.update_audio_segment(AudioSegmentUpdate(**data1))

        # create new entry fro file2
        self.repo.create_audio_segment(
            original_data["project_id"],
            absolute_split_point,
            original_data["end"],
            text2,
            file_path2
        )

        # delete the old audio
        os.remove(file_path)

        return {
            "success": True,
            "message": "Your audio has been split successfully"
        }

    def merge_audios(self, audio_segment_id_1, audio_segment_id_2):
        '''
        audio_segment_id_1: id of the first audio segment in the database
        audio_segment_id_2: id of the second audio segment in the database

        Return

        '''
        # get audio path info from the database
        audio_data_1 = self.repo.get_audio_segment(audio_segment_id_1)
        file_path_1 = audio_data_1["original_audio_path"]
        start1 = audio_data_1["start"]
        end1 = audio_data_1["end"]
        # start1, end1 = self.get_time_range_from_filename(file_path_1)

        audio_data_2 = self.repo.get_audio_segment(audio_segment_id_2)
        file_path_2 = audio_data_2["original_audio_path"]
        start2 = audio_data_2["start"]
        end2 = audio_data_2["end"]
        # start2, end2 = self.get_time_range_from_filename(file_path_2)

        # check if audio 1 precedes audio 2
        start = start1
        end = end2
        if start1 > start2:
            # swap position if it is not
            temp_data = audio_data_2
            temp_file = file_path_2

            audio_data_2 = audio_data_1
            file_path_2 = file_path_1

            audio_data_1 = temp_data
            file_path_1 = temp_file

            start = start2
            end = end1

        # merge 2 audios using ffmpeg
        root_dir = file_path_1.replace(os.path.basename(file_path_1), "")
        filename_out = f"f_{int(start)}_{int(end)}.wav"
        # full_file_path = os.path.join(root_dir, "full_audio.wav")
        file_path_out = os.path.join(root_dir, filename_out)
        os.system(f"ffmpeg -y -i {file_path_1} -i {file_path_2} -filter_complex '[0:a][1:a]concat=n=2:v=0:a=1' {file_path_out}")
        # os.system(f"ffmpeg -y -ss {start/1000.:.10f} -t {end/1000.:.10f} -i {full_file_path} {file_path_out}")

        # concat text from the two audios
        text1 = audio_data_1["text"].strip()
        text2 = audio_data_2["text"].strip()

        if text1 == self.inaudiable_text:
            text1 = "_"

        if text2 == self.inaudiable_text:
            text2 = "_"

        text =  text1 + " " + text2

        # delete audio 2 info from the database
        self.repo.delete_audio_segment(audio_segment_id_2)

        # update audio 1 info in the database to the merged audio
        new_data = {
            "id": audio_segment_id_1,
            "start": start,
            "end": end,
            "text": text,
            "original_audio_path": file_path_out,
            "generated_audio_path": "undefined"
        }
        self.repo.update_audio_segment(AudioSegmentUpdate(**new_data))

        # remove the 2 old audios from the system
        os.remove(file_path_1)
        os.remove(file_path_2)

        return {
            "status_code": 200,
            "message": "Your audios have been merged successfully"
        }

    def generate_audio(self, audio_segment_id, prev_percentage=-1, total_audio_length=1, project_id=-1):
        '''
        audio_segment_id: id of the audio segment in the database

        Generate AI-spoken audio for the given audio segment
        '''

        print(f"\n========= Generating audio segment - {audio_segment_id} ===========")
        # get original audio text info from the database
        audio_data = self.repo.get_audio_segment(audio_segment_id)
        text = audio_data["text"]
        if text == self.inaudiable_text:
            text = "__"

        # get voice from the project
        project_data = self.project_repo.get_audio_project(audio_data["project_id"])
        voice = project_data["voice"]

        original_audio_duration = get_audio_length(audio_data["original_audio_path"])
        new_percentage = 0
        original_audio_duration_ms = original_audio_duration * 1000
        if prev_percentage != -1:
            try:
                new_percentage = int(prev_percentage + original_audio_duration_ms / total_audio_length * 99)
                self.project_repo.update_audio_project(AudioProjectUpdate(**{
                    "id": project_id,
                    "percentage": prev_percentage,
                    "processing_length": original_audio_duration_ms
                }))
            except Exception as e:
                print("==== generate audio exception - {audio_segment_id} =======")
                print(e)
                self.project_repo.update_audio_project(AudioProjectUpdate(**{
                    "id": project_id,
                    "status": Config.project_idle_key,
                    "percentage": 100,
                    "processing_length": 0
                }))
                return HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                                     detail="Lỗi không xác định")

        if not audio_data["has_modified"]:
            return {
                "status_code": 200,
                "generated_audio_path": audio_data["generated_audio_path"],
                "percentage": new_percentage
            }

        # call VBee to generate the AI audio
        data = {
            "content": text,
            "voice": voice
        }
        res = requests.post(f"{self.vbee_api_endpoint}/backend/audio/create", json.dumps(data))
        response_json = res.json()
        print(response_json)

        if response_json["code"] != 200 or response_json["data"] is None:
            return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                 detail="Text-to-speech service is currently not available")

        # speed up the audio if its length is greater than the original audio length
        generated_audio_path = self.get_generated_audio_path(response_json["data"])
        trim_generated_audio_path = audio_data["original_audio_path"].replace(self.original_audio_dir, self.trim_generated_audio_dir)
        final_generated_audio_path = audio_data["original_audio_path"].replace(self.original_audio_dir, self.generated_audio_dir)

        # trim leading and trailing silience
        os.system(f'ffmpeg -y -i {generated_audio_path} -af "silenceremove=start_periods=1:start_duration=0.1:start_threshold=-60dB:detection=peak,aformat=dblp,areverse,silenceremove=start_periods=0.1:start_duration=1:start_threshold=-60dB:detection=peak,aformat=dblp,areverse" {trim_generated_audio_path}')

        generated_audio_duration = get_audio_length(trim_generated_audio_path)

        if generated_audio_duration > original_audio_duration:
            ratio = generated_audio_duration/original_audio_duration
            os.system(f'ffmpeg -y -i {trim_generated_audio_path} -filter:a "atempo={ratio}" -vn {final_generated_audio_path}')
        else:
            os.system(f"cp {trim_generated_audio_path} {final_generated_audio_path}")

        # update the generated path to the database
        new_data = {
            "id": audio_segment_id,
            "generated_audio_path": final_generated_audio_path,
            "has_modified": False
        }
        self.repo.update_audio_segment(AudioSegmentUpdate(**new_data))

        return {
            "status_code": 200,
            "generated_audio_path": final_generated_audio_path,
            "percentage": new_percentage
        }

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
        return self.repo.get_generated_audio_path(audio_id)
