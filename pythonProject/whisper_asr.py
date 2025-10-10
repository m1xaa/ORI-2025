import os
import whisper
import tempfile
import subprocess


def transcribe_video(video_path: str, model_size: str, language: str, start_time: float = 0.0, end_time: float = None):
    if end_time is not None:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            output_path = tmp.name
        duration = end_time - start_time
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-ss", str(start_time),
            "-t", str(duration),
            "-c", "copy", output_path,
            "-loglevel", "error"
        ]
        subprocess.run(cmd, check=True)
        clip_path = output_path
    else:
        clip_path = video_path

    model = whisper.load_model(model_size)
    result = model.transcribe(clip_path, language=language)
    segments = [{
        "start": float(seg["start"]) + start_time,
        "end": float(seg["end"]) + start_time,
        "text": seg["text"].strip()
    } for seg in result["segments"]]

    if end_time is not None and os.path.exists(clip_path):
        os.remove(clip_path)

    return segments
