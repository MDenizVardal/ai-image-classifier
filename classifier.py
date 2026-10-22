import os
import sys
import json
import pickle
import argparse
from pathlib import Path
import numpy as np
from core.features import FeatureExtractor

DATABASE_FILE = "reference_features.pkl"
MIN_CONFIDENCE_THRESHOLD = 40

WEIGHTS = {
    'phash': 0.35,
    'ahash': 0.25,
    'dhash': 0.20,
    'histogram': 0.20
}


class ImageClassifier:
    
    def __init__(self, database_path: str = DATABASE_FILE):
        self.extractor = FeatureExtractor(hash_size=16)
        self.database = self._load_database(database_path)
        self._print_db_info()
        
    def _load_database(self, path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Database not found: {path}\n"
                "Run 'python build_database.py' first to index reference images."
            )
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def _print_db_info(self):
        num_images = len(self.database['images'])
        num_categories = len(self.database['categories'])
        print(f"📚 Loaded database: {num_images} images in {num_categories} categories")
    
    def _hash_similarity(self, hash1: str, hash2: str, hash_size: int = 16) -> float:
        try:
            distance = self.extractor.compute_hash_distance(hash1, hash2)
            max_distance = hash_size * hash_size
            similarity = (1 - (distance / max_distance)) * 100
            return max(0, similarity)
        except Exception:
            return 0.0
    
    def _histogram_similarity(self, hist1: list, hist2: list) -> float:
        try:
            similarity = self.extractor.compute_histogram_similarity(hist1, hist2)
            return min(100, similarity * 100)
        except Exception:
            return 0.0
    
    def _compute_similarity(self, features1: dict, features2: dict) -> float:
        scores = {}
        scores['phash'] = self._hash_similarity(
            features1.get('phash', ''), 
            features2.get('phash', '')
        )
        scores['ahash'] = self._hash_similarity(
            features1.get('ahash', ''), 
            features2.get('ahash', '')
        )
        scores['dhash'] = self._hash_similarity(
            features1.get('dhash', ''), 
            features2.get('dhash', '')
        )
        scores['histogram'] = self._histogram_similarity(
            features1.get('color_histogram', []),
            features2.get('color_histogram', [])
        )
        total_score = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)
        return total_score
    
    def classify(self, image_path: str) -> dict:
        if not os.path.exists(image_path):
            return {
                "matched_category": "error",
                "confidence_score": 0,
                "short_description": f"Image file not found: {image_path}"
            }
        
        print(f"🔍 Analyzing: {Path(image_path).name}")
        
        try:
            input_features = self.extractor.extract_all_features(image_path)
        except Exception as e:
            return {
                "matched_category": "error",
                "confidence_score": 0,
                "short_description": f"Failed to process image: {str(e)}"
            }
        
        category_scores = {}
        best_matches = {}
        
        for ref_image in self.database['images']:
            category = ref_image['category']
            ref_features = ref_image['features']
            similarity = self._compute_similarity(input_features, ref_features)
            
            if category not in category_scores:
                category_scores[category] = []
                best_matches[category] = {'score': 0, 'filename': ''}
            
            category_scores[category].append(similarity)
            
            if similarity > best_matches[category]['score']:
                best_matches[category] = {
                    'score': similarity,
                    'filename': ref_image['filename']
                }
        
        final_scores = {}
        for category, scores in category_scores.items():
            sorted_scores = sorted(scores, reverse=True)
            top_scores = sorted_scores[:3]
            final_scores[category] = np.mean(top_scores)
        
        if not final_scores:
            return {
                "matched_category": "unknown",
                "confidence_score": 0,
                "short_description": "No reference images in database"
            }
        
        best_category = max(final_scores, key=final_scores.get)
        confidence = round(final_scores[best_category], 1)
        
        if confidence < MIN_CONFIDENCE_THRESHOLD:
            return {
                "matched_category": "unknown",
                "confidence_score": confidence,
                "short_description": f"No confident match found. Closest was '{best_category}' at {confidence}%"
            }
        
        best_match_file = best_matches[best_category]['filename']
        num_refs = len(category_scores[best_category])
        
        description = self._generate_description(
            best_category, 
            confidence, 
            best_match_file,
            num_refs
        )
        
        return {
            "matched_category": best_category,
            "confidence_score": confidence,
            "short_description": description
        }
    
    def _generate_description(self, category: str, confidence: float, best_match: str, num_refs: int) -> str:
        confidence_level = (
            "very high" if confidence >= 85 else
            "high" if confidence >= 70 else
            "moderate" if confidence >= 55 else
            "low"
        )
        return (
            f"Image matches '{category}' category with {confidence_level} confidence. "
            f"Best match: {best_match}. Compared against {num_refs} reference images."
        )
    
    def get_all_scores(self, image_path: str) -> dict:
        result = self.classify(image_path)
        
        if result.get("matched_category") == "error":
            return result
        
        try:
            input_features = self.extractor.extract_all_features(image_path)
        except Exception:
            return result
        
        category_scores = {}
        for ref_image in self.database['images']:
            category = ref_image['category']
            similarity = self._compute_similarity(input_features, ref_image['features'])
            
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(similarity)
        
        all_scores = {}
        for category, scores in category_scores.items():
            sorted_scores = sorted(scores, reverse=True)
            all_scores[category] = round(np.mean(sorted_scores[:3]), 1)
        
        result['all_category_scores'] = dict(sorted(
            all_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        return result


def interactive_mode(classifier: ImageClassifier):
    print("\n" + "=" * 60)
    print("🖼️  IMAGE CLASSIFIER - Interactive Mode")
    print("=" * 60)
    print("Enter image path to classify, or 'quit' to exit.")
    print("Tip: You can drag & drop image files into the terminal.\n")
    
    while True:
        try:
            user_input = input("📷 Image path: ").strip().strip('"').strip("'")
            
            if user_input.lower() in ('quit', 'exit', 'q'):
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            result = classifier.get_all_scores(user_input)
            
            print("\n" + "-" * 40)
            print("📊 RESULT:")
            print("-" * 40)
            print(json.dumps(result, indent=2))
            print("-" * 40 + "\n")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Classify images against reference database")
    parser.add_argument("image_path", nargs="?", default=None, help="Path to the image to classify")
    parser.add_argument("--detailed", "-d", action="store_true", help="Show scores for all categories")
    parser.add_argument("--pretty", "-p", action="store_true", help="Pretty-print JSON output")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    try:
        classifier = ImageClassifier()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\n💡 Run 'python build_database.py' first to create the reference database.")
        sys.exit(1)
    
    if args.interactive or args.image_path is None:
        interactive_mode(classifier)
        return
    
    if args.detailed:
        result = classifier.get_all_scores(args.image_path)
    else:
        result = classifier.classify(args.image_path)
    
    indent = 2 if args.pretty else None
    print("\n" + json.dumps(result, indent=indent))


if __name__ == "__main__":
    main()
