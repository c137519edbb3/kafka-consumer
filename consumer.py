import time
import json
import firebase_admin
import threading
import requests
from kafka import KafkaConsumer
from firebase_admin import credentials, firestore 
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from models import CLIPInferenceModel

class MLConsumer:
    def __init__(
        self, topic_name, model_class=CLIPInferenceModel, model_kwargs={},
        kafka_bootstrap_servers='localhost:9092', backend_url="http://backend-server/api/get-classes"
    ):
        self.consumer = KafkaConsumer(
            topic_name,
            bootstrap_servers=kafka_bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )

        cred = credentials.Certificate("./key.json")
        firebase_admin.initialize_app(cred)

        self.db = firestore.client()
        self.db_ref = self.db.collection("organizations")

        self.model = model_class(**model_kwargs)

        # Thread pool for batch writes
        self.executor = ThreadPoolExecutor(max_workers=5)

        # Log queue for batch writes
        self.log_queue = Queue()
        self.BATCH_SIZE = 3

        # Local class cache per camera_id
        self.class_cache = {}

        # Backend API URL for fetching classes
        self.backend_url = backend_url

        # Start the background cache updater
        self.cache_update_thread = threading.Thread(target=self.update_cache_periodically, daemon=True)
        self.cache_update_thread.start()

        # Background writer thread
        self.writer_thread = threading.Thread(target=self.batch_write_logs, daemon=True)
        self.writer_thread.start()

    def update_cache_periodically(self, interval=30):
        """Fetch updated classes from the backend and refresh the cache periodically"""
        while True:
            try:
                response = requests.get(self.backend_url)
                if response.status_code == 200:
                    data = response.json()
                    
                    for camera_id, classes in data.items():
                        self.class_cache[camera_id] = {
                            "good_classes": set(classes["good_classes"]),
                            "bad_classes": set(classes["bad_classes"])
                        }

                    print(f"✅ Cache updated at {time.strftime('%H:%M:%S')} | Cameras: {len(self.class_cache)}")
                else:
                    print(f"⚠️ Failed to update cache, status: {response.status_code}")
            except Exception as e:
                print(f"❌ Cache update error: {e}")

            time.sleep(interval)

    def enqueue_log(self, organization_id, camera_id, log_data):
        """Queue log for batch writing"""
        self.log_queue.put((organization_id, camera_id, log_data))

    def batch_write_logs(self):
        """Background thread: Batch Firestore writes for efficiency"""
        while True:
            batch_data = []

            while not self.log_queue.empty() and len(batch_data) < self.BATCH_SIZE:
                organization_id, camera_id, log_data = self.log_queue.get()
                batch_data.append((organization_id, camera_id, log_data))

            if batch_data:
                batch = self.db.batch()

                for org_id, cam_id, log_data in batch_data:
                    log_ref = (
                        self.db_ref.document(org_id)
                        .collection("cameras")
                        .document(cam_id)
                        .collection("logs")
                        .document()
                    )
                    batch.set(log_ref, log_data)
                    batch.commit()
                    print(f"🔥 Log pushed to Firestore with ID: {log_ref.id}")

                print(f"🔥 Batch committed with {len(batch_data)} logs.")

            time.sleep(1)

    def process_stream(self):
        try:
            print("Starting to process stream...")
            for message in self.consumer:
                frame_data = message.value['frame']
                organization_id = message.value['organization_id']
                camera_id = message.value['camera_id']
                timestamp = message.value['timestamp']

                # Fetch classes for the camera from cache (O(1) lookup)
                camera_classes = self.class_cache.get(camera_id, {"good_classes": set(), "bad_classes": set()})

                predictions = self.model.predict(frame_data)
                best_label, best_confidence = max(predictions, key=lambda x: x[1])

                if best_label in camera_classes["bad_classes"]:
                    category = "BAD CLASS ⚠️"
                    log_entry = {
                        "timestamp": timestamp,
                        "event": best_label,
                        "confidence": best_confidence,
                        "camera_id": camera_id,
                        "organization_id": organization_id,
                    }
                    self.enqueue_log(organization_id, camera_id, log_entry)

                elif best_label in camera_classes["good_classes"]:
                    category = "GOOD CLASS ✅"
                else:
                    category = "UNKNOWN"

                print(f"\nTimestamp: {message.value['timestamp']}")
                print(f"{best_label} ({best_confidence*100:.2f}%) - {category}")
                print("-------------------")

        except KeyboardInterrupt:
            print("Stopping consumer...")
        finally:
            self.consumer.close()


if __name__ == "__main__":
    consumer = MLConsumer(
        topic_name='camera_feed',
        model_kwargs={'model_name': 'openai/clip-vit-large-patch14'},
        backend_url="http://backend-server/api/get-classes"
    )

    consumer.process_stream()
