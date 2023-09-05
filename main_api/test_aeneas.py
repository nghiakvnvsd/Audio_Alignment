from aeneas.executetask import ExecuteTask
from aeneas.task import Task

# create Task object
config_string = u"task_language=vie|is_text_type=plain|os_task_file_format=json"
task = Task(config_string=config_string)
task.audio_file_path_absolute = "./inputs/sample1_trim2.wav"
task.text_file_path_absolute = "./inputs/sample1_trim2_word.txt"
task.sync_map_file_path_absolute = "./inputs/sample1_trim2.json"

# process Task
ExecuteTask(task).execute()

# output sync map to file
task.output_sync_map_file()
