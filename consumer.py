import queue
import time
import json
from auth import Auth
from config import Config
import firebase_admin
import threading
import requests
from kafka import KafkaConsumer
from firebase_admin import credentials, firestore 
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from models import CLIPInferenceModel

class MLConsumer:
    def __init__(self, topic_name, model_class=CLIPInferenceModel, model_kwargs={}, 
                 kafka_bootstrap_servers=Config.KAFKA_BOOTSTRAP_SERVER_URL, auth=None):
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
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.log_queue = Queue()
        self.BATCH_SIZE = 3

        self.auth = auth
        self.camera_manager = auth.camera_manager if auth else None

        self.writer_thread = threading.Thread(target=self.batch_write_logs, daemon=True)
        self.writer_thread.start()

    def enqueue_log(self, organization_id, camera_id, log_data):
        """Queue log for batch writing."""
        try:
            self.log_queue.put((organization_id, camera_id, log_data))
        except Exception as e:
            print(f"❌ Failed to enqueue log: {str(e)}")

    def process_stream(self):
        """Processes Kafka stream data with new data structure"""
        try:
            print("Starting to process stream...")
            for message in self.consumer:
                print(f"Received message: {message.value}")
                if self.camera_manager and self.camera_manager.needs_sync():
                    self.auth.sync_cache()  # This will update camera_manager

                frame_data = message.value["frame"]
                organization_id = message.value["organization_id"]
                camera_id = message.value["camera_id"]
                timestamp = message.value["timestamp"]

                # Get classes using new data manager
                camera_classes = self.camera_manager.get_clip_classes(camera_id)
                all_labels = list(camera_classes["good_classes"]) + list(camera_classes["bad_classes"])
                self.model.set_labels(all_labels)

                predictions = self.model.predict(frame_data)
                best_label, best_confidence = max(predictions, key=lambda x: x[1])

                if best_label in camera_classes["bad_classes"]:
                    category = "BAD CLASS ⚠️"
                    # Get anomaly details for logging
                    anomaly = self.camera_manager.get_anomaly_details(camera_id, best_label)
                    if anomaly:
                        log_entry = {
                            "timestamp": timestamp,
                            "event": best_label,
                            "confidence": best_confidence,
                            "camera_id": int(camera_id),
                            "organization_id": int(organization_id),
                            "anomaly_id": anomaly.anomaly_id,
                            "criticality": anomaly.criticality
                        }
                        self.enqueue_log(organization_id, camera_id, log_entry)

                elif best_label in camera_classes["good_classes"]:
                    category = "GOOD CLASS ✅"
                else:
                    category = "UNKNOWN"

                print(f"\nTimestamp: {timestamp}")
                print(f"{best_label} ({best_confidence*100:.2f}%) - {category}")
                print("-------------------")

        except KeyboardInterrupt:
            print("Stopping consumer...")
        finally:
            self.consumer.close()

    def batch_write_logs(self):
        """
        Batch writes logs to Firestore with proper anomaly information.
        Uses batching for better performance and transaction safety.
        """
        while True:
            batch_data = []
            batch = self.db.batch()  # Create a new batch
            
            # Collect logs up to BATCH_SIZE
            while not self.log_queue.empty() and len(batch_data) < self.BATCH_SIZE:
                try:
                    organization_id, camera_id, log_data = self.log_queue.get_nowait()
                    
                    # Validate required fields
                    if not all([organization_id, camera_id, log_data]):
                        print(f"❌ Invalid log data: org_id={organization_id}, camera_id={camera_id}")
                        continue

                    # Ensure all fields are properly formatted
                    log_data.update({
                        "timestamp": log_data.get("timestamp", time.time()),
                        "camera_id": int(camera_id),
                        "organization_id": int(organization_id),
                    })

                    # Add to batch data
                    batch_data.append((organization_id, camera_id, log_data))
                    
                except queue.Empty:
                    break  # Queue is empty
                except Exception as e:
                    print(f"❌ Error processing log entry: {str(e)}")
                    continue

            # Process batch if we have data
            if batch_data:
                try:
                    # Add each log to the batch
                    for org_id, cam_id, log_data in batch_data:
                        log_ref = (
                            self.db_ref
                            .document(str(org_id))
                            .collection("cameras")
                            .document(str(cam_id))
                            .collection("logs")
                            .document()  # Auto-generate document ID
                        )
                        batch.set(log_ref, log_data)

                    # Commit the batch
                    batch.commit()
                    print(f"✅ Wrote {len(batch_data)} logs to Firestore")
                    
                except Exception as e:
                    print(f"🔥 Firestore batch write failed: {str(e)}")
                    
                    # Attempt to requeue failed logs
                    for org_id, cam_id, log_data in batch_data:
                        try:
                            self.log_queue.put((org_id, cam_id, log_data))
                            print(f"⚠️ Requeued failed log for camera {cam_id}")
                        except Exception as requeue_error:
                            print(f"❌ Failed to requeue log: {str(requeue_error)}")

            # Sleep to prevent CPU overuse
            time.sleep(1)



if __name__ == "__main__":
    auth = Auth()
    if not auth.login():
        print("Failed to authenticate with EyeconAI API")
        exit(1)

    cameras_data = auth.sync_cache()
    if not cameras_data:
        print("No online cameras found")
        exit(1)

    print("Camera data loaded:", cameras_data)

    consumer = MLConsumer(
        topic_name="camera_feed",
        model_kwargs={"model_name": "openai/clip-vit-large-patch14"},
        auth=auth
    )

    consumer.process_stream()
