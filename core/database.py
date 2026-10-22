import sqlite3
import os
import json
from datetime import datetime
from config import DATABASE_PATH


class Database:
    
    def __init__(self, db_path=None):
        self.db_path = db_path or DATABASE_PATH
        self._ensure_data_directory()
        self._init_database()
    
    def _ensure_data_directory(self):
        data_dir = os.path.dirname(self.db_path)
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def _init_database(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    original_name TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    width INTEGER,
                    height INTEGER,
                    format TEXT,
                    phash TEXT,
                    ahash TEXT,
                    dhash TEXT,
                    color_histogram TEXT,
                    dominant_colors TEXT,
                    tags TEXT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_phash ON images(phash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ahash ON images(ahash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_dhash ON images(dhash)')
            conn.commit()
    
    def _get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def add_image(self, image_data: dict) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO images (
                    filename, original_name, file_path, file_size,
                    width, height, format, phash, ahash, dhash,
                    color_histogram, dominant_colors, tags, description
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                image_data.get('filename'),
                image_data.get('original_name'),
                image_data.get('file_path'),
                image_data.get('file_size'),
                image_data.get('width'),
                image_data.get('height'),
                image_data.get('format'),
                image_data.get('phash'),
                image_data.get('ahash'),
                image_data.get('dhash'),
                json.dumps(image_data.get('color_histogram', [])),
                json.dumps(image_data.get('dominant_colors', [])),
                json.dumps(image_data.get('tags', [])),
                image_data.get('description', '')
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_image(self, image_id: int) -> dict:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM images WHERE id = ?', (image_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_dict(row)
            return None
    
    def get_all_images(self) -> list:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM images ORDER BY created_at DESC')
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
    
    def get_all_hashes(self) -> list:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, filename, phash, ahash, dhash, color_histogram, dominant_colors
                FROM images
            ''')
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
    
    def delete_image(self, image_id: int) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM images WHERE id = ?', (image_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def update_tags(self, image_id: int, tags: list) -> bool:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE images 
                SET tags = ?, updated_at = CURRENT_TIMESTAMP 
                WHERE id = ?
            ''', (json.dumps(tags), image_id))
            conn.commit()
            return cursor.rowcount > 0
    
    def search_by_tag(self, tag: str) -> list:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM images 
                WHERE tags LIKE ? 
                ORDER BY created_at DESC
            ''', (f'%"{tag}"%',))
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
    
    def get_image_count(self) -> int:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM images')
            return cursor.fetchone()[0]
    
    def _row_to_dict(self, row) -> dict:
        d = dict(row)
        for field in ['color_histogram', 'dominant_colors', 'tags']:
            if field in d and d[field]:
                try:
                    d[field] = json.loads(d[field])
                except json.JSONDecodeError:
                    d[field] = []
        return d
