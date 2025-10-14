import whisper


def transcribe_video(video_path: str, model_size: str, language: str):
    model = whisper.load_model(model_size)
    result = model.transcribe(video_path, language=language)
    segments = []
    for seg in result["segments"]:
        segments.append({
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": seg["text"].strip()
        })

    return segments
