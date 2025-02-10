from dataclasses import dataclass
from typing import List, Dict, Set
import time

@dataclass
class Anomaly:
    anomaly_id: int
    description: str
    criticality: str

@dataclass
class Camera:
    camera_id: int
    normal_conditions: Set[str]
    anomalies: Dict[str, Anomaly]

class CameraDataManager:
    def __init__(self):
        self.cameras: Dict[int, Camera] = {}
        self._all_classes_cache: Dict[int, Dict[str, set]] = {}
        self.last_sync: float = 0
        self.SYNC_INTERVAL = 300  # 5 minutes

    def load_from_api_response(self, api_data: dict) -> None:
        """Loads camera data from new API response format"""
        for camera_data in api_data.get("cameras", []):
            camera_id = camera_data["cameraId"]
            
            normal_conditions = {
                condition["description"] 
                for condition in camera_data.get("normalConditions", [])
            }
            
            anomalies = {
                anomaly["description"]: Anomaly(
                    anomaly_id=anomaly["anomalyId"],
                    description=anomaly["description"],
                    criticality=anomaly["criticality"]
                )
                for anomaly in camera_data.get("anomalies", [])
            }
            
            self.cameras[camera_id] = Camera(
                camera_id=camera_id,
                normal_conditions=normal_conditions,
                anomalies=anomalies
            )
            
            self._update_clip_classes_cache(camera_id)

    def _update_clip_classes_cache(self, camera_id: int) -> None:
        """Updates the CLIP classes cache for faster lookups"""
        camera = self.cameras.get(camera_id)
        if camera:
            self._all_classes_cache[camera_id] = {
                "good_classes": camera.normal_conditions,
                "bad_classes": set(camera.anomalies.keys())
            }

    def get_clip_classes(self, camera_id: int) -> Dict[str, set]:
        """Gets CLIP classes for a camera"""
        return self._all_classes_cache.get(camera_id, {"good_classes": set(), "bad_classes": set()})

    def get_anomaly_details(self, camera_id: int, description: str) -> Anomaly:
        """Gets anomaly details if it exists"""
        camera = self.cameras.get(camera_id)
        if camera:
            return camera.anomalies.get(description)
        return None

    def needs_sync(self) -> bool:
        """Checks if data needs to be synced"""
        return time.time() - self.last_sync > self.SYNC_INTERVAL

    def update_last_sync(self) -> None:
        """Updates last sync timestamp"""
        self.last_sync = time.time()