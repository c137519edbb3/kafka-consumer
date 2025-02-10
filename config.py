from dotenv import load_dotenv
import os

load_dotenv()

class Config:
    KAFKA_BOOTSTRAP_SERVER_URL = os.getenv('KAFKA_BOOTSTRAP_SERVER_URL')
    EYECONAI_API_URL = os.getenv('EYECONAI_API_URL')
    EYECONAI_USERNAME = os.getenv('EYECONAI_USERNAME')
    EYECONAI_PASSWORD = os.getenv('EYECONAI_PASSWORD')
    