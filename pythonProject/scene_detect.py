from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


def detect_scenes(video_path: str, threshold: float = 40.0):
    video = open_video(video_path)

    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list()

    scenes = []
    for i, (start, end) in enumerate(scene_list):
        scenes.append({
            "id": i + 1,
            "start": start.get_seconds(),
            "end": end.get_seconds()
        })
    return scenes
