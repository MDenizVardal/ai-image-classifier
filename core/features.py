import numpy as np
from PIL import Image
import imagehash
from collections import Counter
import cv2
from config import HASH_SIZE


class FeatureExtractor:
    
    def __init__(self, hash_size=None):
        self.hash_size = hash_size or HASH_SIZE
    
    def extract_all_features(self, image_path: str) -> dict:
        image = Image.open(image_path)
        width, height = image.size
        format_type = image.format or image_path.split('.')[-1].upper()
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        features = {
            'width': width,
            'height': height,
            'format': format_type,
            'phash': self.compute_phash(image),
            'ahash': self.compute_ahash(image),
            'dhash': self.compute_dhash(image),
            'color_histogram': self.compute_color_histogram(image),
            'dominant_colors': self.extract_dominant_colors(image_path),
        }
        return features
    
    def compute_phash(self, image: Image.Image) -> str:
        return str(imagehash.phash(image, hash_size=self.hash_size))
    
    def compute_ahash(self, image: Image.Image) -> str:
        return str(imagehash.average_hash(image, hash_size=self.hash_size))
    
    def compute_dhash(self, image: Image.Image) -> str:
        return str(imagehash.dhash(image, hash_size=self.hash_size))
    
    def compute_color_histogram(self, image: Image.Image, bins: int = 16) -> list:
        image = image.resize((256, 256))
        img_array = np.array(image)
        
        histograms = []
        for i in range(3):
            hist, _ = np.histogram(img_array[:, :, i], bins=bins, range=(0, 256))
            hist = hist.astype(float) / hist.sum()
            histograms.extend(hist.tolist())
        
        return histograms
    
    def extract_dominant_colors(self, image_path: str, k: int = 5) -> list:
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (100, 100))
        pixels = img.reshape(-1, 3).astype(np.float32)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        label_counts = Counter(labels.flatten())
        total_pixels = len(labels)
        
        colors = []
        for i, center in enumerate(centers):
            percentage = (label_counts[i] / total_pixels) * 100
            colors.append({
                'rgb': [int(c) for c in center],
                'hex': '#{:02x}{:02x}{:02x}'.format(int(center[0]), int(center[1]), int(center[2])),
                'percentage': round(percentage, 2)
            })
        
        colors.sort(key=lambda x: x['percentage'], reverse=True)
        return colors
    
    def compute_hash_distance(self, hash1: str, hash2: str) -> int:
        h1 = imagehash.hex_to_hash(hash1)
        h2 = imagehash.hex_to_hash(hash2)
        return h1 - h2
    
    def compute_histogram_similarity(self, hist1: list, hist2: list) -> float:
        if not hist1 or not hist2:
            return 0.0
        
        h1 = np.array(hist1)
        h2 = np.array(hist2)
        intersection = np.minimum(h1, h2).sum()
        return float(intersection)
