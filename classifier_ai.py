import os
import sys
import json
import pickle
import argparse
from pathlib import Path
import numpy as np

try:
    import torch
    import torchvision.transforms as transforms
    from torchvision import models
    from PIL import Image
except ImportError:
    print("❌ Missing required packages!")
    print("Run: pip install torch torchvision pillow")
    sys.exit(1)

from config import REFERENCE_FOLDER

DATABASE_FILE = "ai_features.pkl"
MIN_CONFIDENCE_THRESHOLD = 40


class AIFeatureExtractor:
    
    def __init__(self):
        print("🧠 Loading AI model (ResNet50)...")
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        if torch.cuda.is_available():
            print(f"   Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("   Using CPU")
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(image_tensor)
        
        features = features.cpu().numpy().flatten()
        features = features / np.linalg.norm(features)
        return features


class AIClassifier:
    
    def __init__(self, database_path: str = DATABASE_FILE):
        self.extractor = AIFeatureExtractor()
        self.database = self._load_database(database_path)
        
    def _load_database(self, path: str) -> dict:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"AI Database not found: {path}\n"
                "Run 'python classifier_ai.py --build' first."
            )
        with open(path, 'rb') as f:
            db = pickle.load(f)
        print(f"📚 Loaded: {len(db['images'])} images in {len(db['categories'])} categories")
        return db
    
    def _cosine_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        return float(np.dot(feat1, feat2))
    
    def classify(self, image_path: str) -> dict:
        if not os.path.exists(image_path):
            return {
                "matched_category": "error",
                "confidence_score": 0,
                "short_description": f"Image file not found: {image_path}"
            }
        
        print(f"🔍 Analyzing: {Path(image_path).name}")
        
        try:
            input_features = self.extractor.extract_features(image_path)
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
            ref_features = np.array(ref_image['features'])
            similarity = self._cosine_similarity(input_features, ref_features)
            similarity_pct = (similarity + 1) * 50
            
            if category not in category_scores:
                category_scores[category] = []
                best_matches[category] = {'score': 0, 'filename': ''}
            
            category_scores[category].append(similarity_pct)
            
            if similarity_pct > best_matches[category]['score']:
                best_matches[category] = {
                    'score': similarity_pct,
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
                "short_description": f"No confident match. Closest: '{best_category}' at {confidence}%"
            }
        
        confidence_level = (
            "very high" if confidence >= 85 else
            "high" if confidence >= 70 else
            "moderate" if confidence >= 55 else
            "low"
        )
        
        return {
            "matched_category": best_category,
            "confidence_score": confidence,
            "short_description": f"Image matches '{best_category}' with {confidence_level} confidence. Best match: {best_matches[best_category]['filename']}"
        }
    
    def get_all_scores(self, image_path: str) -> dict:
        result = self.classify(image_path)
        
        if result.get("matched_category") == "error":
            return result
        
        try:
            input_features = self.extractor.extract_features(image_path)
        except Exception:
            return result
        
        category_scores = {}
        for ref_image in self.database['images']:
            category = ref_image['category']
            ref_features = np.array(ref_image['features'])
            similarity = self._cosine_similarity(input_features, ref_features)
            similarity_pct = (similarity + 1) * 50
            
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(similarity_pct)
        
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


def build_ai_database():
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.avif', '.gif', '.bmp'}
    
    extractor = AIFeatureExtractor()
    database = {'categories': {}, 'images': []}
    
    reference_path = Path(REFERENCE_FOLDER)
    
    if not reference_path.exists():
        print(f"❌ Reference folder not found: {REFERENCE_FOLDER}")
        return None
    
    print(f"\n📂 Scanning: {REFERENCE_FOLDER}")
    print("-" * 60)
    
    total_images = 0
    
    for category_folder in sorted(reference_path.iterdir()):
        if not category_folder.is_dir():
            continue
        
        category_name = category_folder.name
        database['categories'][category_name] = []
        image_count = 0
        
        for image_file in sorted(category_folder.iterdir()):
            if image_file.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            
            try:
                print(f"  [{category_name}] {image_file.name}...", end=" ", flush=True)
                features = extractor.extract_features(str(image_file))
                
                image_data = {
                    'category': category_name,
                    'filename': image_file.name,
                    'path': str(image_file),
                    'features': features.tolist()
                }
                
                database['images'].append(image_data)
                database['categories'][category_name].append(len(database['images']) - 1)
                image_count += 1
                print("✓")
                
            except Exception as e:
                print(f"✗ Error: {e}")
        
        print(f"  └── {category_name}: {image_count} images")
        total_images += image_count
    
    with open(DATABASE_FILE, 'wb') as f:
        pickle.dump(database, f)
    
    print("-" * 60)
    print(f"✅ AI Database saved to: {DATABASE_FILE}")
    print(f"📊 Total: {total_images} images in {len(database['categories'])} categories")
    
    return database


def interactive_mode(classifier: AIClassifier):
    print("\n" + "=" * 60)
    print("🤖  AI IMAGE CLASSIFIER - Interactive Mode")
    print("=" * 60)
    print("Enter image path to classify, or 'quit' to exit.")
    print("Tip: Drag & drop image files into the terminal.\n")
    
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
    parser = argparse.ArgumentParser(description="AI Image Classifier")
    parser.add_argument("image_path", nargs="?", default=None, help="Image to classify")
    parser.add_argument("--build", "-b", action="store_true", help="Build AI database")
    parser.add_argument("--detailed", "-d", action="store_true", help="Show all scores")
    parser.add_argument("--pretty", "-p", action="store_true", help="Pretty JSON output")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    if args.build:
        build_ai_database()
        return
    
    try:
        classifier = AIClassifier()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("\n💡 Run 'python classifier_ai.py --build' first.")
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
