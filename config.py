import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024
DATABASE_PATH = os.path.join(BASE_DIR, 'data', 'pictures.db')
HASH_SIZE = 16
THUMBNAIL_SIZE = (256, 256)
SIMILARITY_THRESHOLD = 0.85
