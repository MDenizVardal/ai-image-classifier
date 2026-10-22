import os
import pickle
from pathlib import Path
from core.features import FeatureExtractor

REFERENCE_FOLDER = r"C:\Users\Deniz\Desktop\ProjectPictureIdent"
DATABASE_FILE = "reference_features.pkl"
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.avif', '.gif', '.bmp'}


def build_reference_database():
    extractor = FeatureExtractor(hash_size=16)
    database = {'categories': {}, 'images': []}
    
    reference_path = Path(REFERENCE_FOLDER)
    
    if not reference_path.exists():
        print(f"❌ Reference folder not found: {REFERENCE_FOLDER}")
        return None
    
    print(f"📂 Scanning reference folder: {REFERENCE_FOLDER}")
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
                print(f"  [{category_name}] {image_file.name}...", end=" ")
                features = extractor.extract_all_features(str(image_file))
                
                image_data = {
                    'category': category_name,
                    'filename': image_file.name,
                    'path': str(image_file),
                    'features': features
                }
                
                database['images'].append(image_data)
                database['categories'][category_name].append(len(database['images']) - 1)
                image_count += 1
                print("✓")
                
            except Exception as e:
                print(f"✗ Error: {e}")
        
        print(f"  └── {category_name}: {image_count} images indexed")
        total_images += image_count
    
    with open(DATABASE_FILE, 'wb') as f:
        pickle.dump(database, f)
    
    print("-" * 60)
    print(f"✅ Database saved to: {DATABASE_FILE}")
    print(f"📊 Total: {total_images} images in {len(database['categories'])} categories")
    
    print("\n📋 Category Summary:")
    for cat_name, indices in database['categories'].items():
        print(f"   • {cat_name}: {len(indices)} images")
    
    return database


if __name__ == "__main__":
    build_reference_database()
