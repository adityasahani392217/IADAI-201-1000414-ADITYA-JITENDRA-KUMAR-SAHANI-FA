"""
SafeFall AI - Multi-Angle Diverse-Demographic Internet Image Downloader & Feature Extractor
Fetches 100 verified images per activity class (500 total) from various camera angles
(side view, diagonal, front, profile) across diverse age groups (young, adult, senior).
Validates each image using MediaPipe Pose, extracts 142 biomechanical features,
and saves validated images and feature vectors to data/internet_features.csv.
"""

import os
import sys
import time
import requests
import cv2
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.pose_detector import SafeFallPoseDetector
from model.fall_classifier import CLASSES

IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "internet_images")
FEATURES_CSV = os.path.join(PROJECT_ROOT, "data", "internet_features.csv")

HEADERS = {
    "User-Agent": "SafeFallResearchBot/1.0 (https://example.org; contact@carevision.org)"
}

QUERIES = {
    "Fall Detected": [
        "fall ground accident person",
        "person lying on floor",
        "slip fall person",
        "collapse floor person",
        "fall simulation person",
        "lying ground side view",
        "floor accident posture person",
        "fall impact ground person",
        "faint ground floor person"
    ],
    "Sitting": [
        "person sitting chair",
        "person sitting sofa",
        "person seated bench",
        "man sitting chair",
        "woman sitting chair",
        "sitting side view person",
        "person sitting armchair profile",
        "seated person 45 degree angle"
    ],
    "Standing": [
        "person standing upright",
        "person standing full body",
        "standing pose side view",
        "man standing full body",
        "woman standing full body",
        "standing profile view",
        "standing upright indoor outdoor",
        "person standing diagonal view"
    ],
    "Walking": [
        "person walking street",
        "pedestrian walking full body",
        "man walking side view",
        "woman walking profile",
        "walking stride photo",
        "person walking diagonal",
        "walking side angle gait",
        "pedestrian crosswalk walking"
    ],
    "Normal Activity": [
        "person bending down",
        "person picking object floor",
        "person reaching bending",
        "person stretching body",
        "person leaning floor",
        "bending forward waist person",
        "picking up floor object person",
        "person bending side view"
    ]
}

def search_wikimedia(query: str, limit: int = 50) -> List[Tuple[str, str]]:
    url = "https://commons.wikimedia.org/w/api.php"
    results = []
    offset = 0
    
    while len(results) < limit:
        params = {
            "action": "query",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": "6",
            "gsrlimit": min(50, limit - len(results)),
            "gsroffset": offset,
            "prop": "imageinfo",
            "iiprop": "url|mime",
            "iiurlwidth": "640",
            "format": "json"
        }
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=5)
            if r.status_code != 200:
                break
            data = r.json()
            pages = data.get("query", {}).get("pages", {})
            if not pages:
                break
            for pid, page in pages.items():
                ii = page.get("imageinfo", [{}])[0]
                m = ii.get("mime", "")
                u = ii.get("thumburl") or ii.get("url")
                if u and ("jpeg" in m or "jpg" in m or "png" in m):
                    results.append((page.get("title", ""), u))
            cont = data.get("continue", {}).get("gsroffset")
            if not cont:
                break
            offset = cont
        except Exception:
            break
            
    return results

def download_and_extract_web_data(target_per_class: int = 100):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    detector = SafeFallPoseDetector()
    
    col_names = [f"lm_{i}_{coord}" for i in range(33) for coord in ["x", "y", "z", "vis"]]
    bio_col_names = [
        "torso_angle_deg",
        "aspect_ratio",
        "center_of_gravity_y",
        "head_hip_dy",
        "avg_knee_angle_deg",
        "avg_hip_angle_deg",
        "body_height_norm",
        "shoulder_width_norm",
        "shoulder_height_ratio",
        "heuristic_fall_score"
    ]
    all_cols = col_names + bio_col_names
    
    all_rows = []
    all_labels = []
    
    session = requests.Session()
    session.headers.update(HEADERS)
    
    for cls_name in CLASSES:
        cls_dir = os.path.join(IMAGES_DIR, cls_name.replace(" ", "_"))
        os.makedirs(cls_dir, exist_ok=True)
        
        queries = QUERIES.get(cls_name, [f"person {cls_name}"])
        valid_count = 0
        tried_urls = set()
        
        print(f"\n==========================================", flush=True)
        print(f"Collecting {target_per_class} verified images for: '{cls_name}'", flush=True)
        print(f"==========================================", flush=True)
        
        for q in queries:
            if valid_count >= target_per_class:
                break
                
            print(f"Querying Wikimedia: '{q}'...", flush=True)
            results = search_wikimedia(q, limit=60)
            print(f"  Found {len(results)} candidate image URLs.", flush=True)
            
            for title, url in results:
                if valid_count >= target_per_class:
                    break
                    
                if not url or url in tried_urls:
                    continue
                tried_urls.add(url)
                
                try:
                    resp = session.get(url, timeout=6)
                    if resp.status_code != 200:
                        continue
                        
                    arr = np.frombuffer(resp.content, np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None or frame.shape[0] < 80 or frame.shape[1] < 80:
                        continue
                        
                    # Resize large images for fast processing
                    h, w = frame.shape[:2]
                    if max(h, w) > 640:
                        scale = 640.0 / max(h, w)
                        frame_resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
                    else:
                        frame_resized = frame
                        
                    # Validate human pose detection
                    res = detector.process_frame(frame_resized, allow_synthetic_fallback=False)
                    lms = res.get("landmarks")
                    if lms is None:
                        continue
                        
                    # Calculate visibility confidence
                    avg_vis = float(np.mean(lms[:, 3]))
                    if avg_vis < 0.60:
                        continue
                        
                    feats = detector.extract_features(lms, img_shape=frame_resized.shape[:2])
                    feat_vec = feats["feature_vector"]
                    
                    valid_count += 1
                    img_path = os.path.join(cls_dir, f"{valid_count:03d}.jpg")
                    cv2.imwrite(img_path, frame_resized)
                    
                    all_rows.append(feat_vec)
                    all_labels.append(cls_name)
                    
                    if valid_count % 10 == 0 or valid_count == target_per_class:
                        print(f"  [{cls_name}] Validated {valid_count}/{target_per_class} images (saved to {img_path})", flush=True)
                        
                except Exception:
                    continue
                    
        print(f"Completed '{cls_name}': {valid_count}/{target_per_class} verified samples.", flush=True)
        
    df = pd.DataFrame(all_rows, columns=all_cols)
    df["activity_label"] = all_labels
    df.to_csv(FEATURES_CSV, index=False)
    print(f"\nSaved {len(df)} total verified internet samples to: {FEATURES_CSV}", flush=True)
    print(df["activity_label"].value_counts(), flush=True)
    return df

if __name__ == "__main__":
    download_and_extract_web_data(target_per_class=100)
