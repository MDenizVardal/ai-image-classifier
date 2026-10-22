# AI Image Classifier

Classifies an image into one of your own categories by comparing it against a folder of reference images. It has two independent classifiers so the classic and deep learning approaches can be compared side by side.

| | Basic classifier | AI classifier |
|---|---|---|
| Features | pHash, aHash, dHash, color histogram | ResNet50 embeddings (ImageNet weights, final layer removed) |
| Similarity | Weighted hash distance + histogram intersection | Cosine similarity |
| Speed | Fast, CPU only | Slower, uses GPU if available |
| Strength | Near-duplicates, same object in similar framing | Same kind of object in different photos |

## How it works

1. **Build a reference database.** Every image in `reference_images/<category>/` is turned into a feature vector and saved to a local `.pkl` file.
2. **Classify.** The input image goes through the same feature extraction and is compared with every reference image. Each category's score is the mean of its 3 best matches, and the best category wins.
3. **Confidence.** Scores below 40 return `unknown`. Above that, the result is labelled low, moderate (55+), high (70+) or very high (85+).

Basic classifier weights: pHash 0.35, aHash 0.25, dHash 0.20, color histogram 0.20.

## Getting started

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Put 5–25 images per category in `reference_images/`, one folder per category:

```
reference_images/
├── cat/
├── dog/
└── watermelon/
```

Then build the databases and classify:

```bash
python build_database.py              # basic classifier
python classifier_ai.py --build       # AI classifier (downloads ResNet50 weights on first run)

python classifier.py photo.jpg -d -p      # basic, with all category scores
python classifier_ai.py photo.jpg -d -p   # AI, with all category scores
python classifier_ai.py -i                # interactive mode
```

On Windows, the `.bat` files run the same steps.

## Example output

```json
{
  "matched_category": "stripes",
  "confidence_score": 95.6,
  "short_description": "Image matches 'stripes' category with very high confidence. Best match: 4.png. Compared against 5 reference images.",
  "all_category_scores": { "stripes": 95.6, "checks": 55.9, "circles": 49.9 }
}
```

## Project structure

```
├── classifier.py        basic classifier (hashing + color)
├── classifier_ai.py     AI classifier (ResNet50) and its database builder
├── build_database.py    database builder for the basic classifier
├── config.py            reference folder and hash size
├── core/features.py     perceptual hashes, color histogram, dominant colors (k-means)
└── docs/                project documentation and presentation
```

Built with Python, PyTorch, TorchVision, OpenCV, Pillow and ImageHash.
