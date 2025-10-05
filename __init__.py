"""
BDDOIA Safe/Unsafe Action Dataset for FiftyOne

A comprehensive COCO-format extension of the BDD-OIA dataset with explicit 
unsafe action labels and reason explanations. Designed for benchmarking 
safety-aware vision-language models in autonomous driving scenarios.
"""

import os
import json
import random
import zipfile
import requests
import fiftyone as fo
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

SPLIT_MAPPING = {
    "train": "train_25k_images.json",
    "validation": "val_25k_images.json", 
    "test": "test_25k_images.json"
}

ACTION_CLASSES = ["forward", "stop", "left", "right"]

REASON_CLASSES = [
    "Traffic light is green",
    "Follow traffic", 
    "Road is clear",
    "Traffic light",
    "Traffic sign",
    "Obstacle: car",
    "Obstacle: person",
    "Obstacle: rider",
    "Obstacle: others",
    "No lane on the left",
    "Obstacles on the left lane",
    "Solid line on the left",
    "On the left-turn lane",
    "Traffic light allows (left)",
    "Front car turning left",
    "No lane on the right",
    "Obstacles on the right lane",
    "Solid line on the right",
    "On the right-turn lane",
    "Traffic light allows (right)",
    "Front car turning right"
]


def _download_file(url: str, filepath: str) -> bool:
    """Download file from URL to filepath."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception:
        return False


def _extract_zip(zip_path: str, extract_dir: str) -> bool:
    """Extract zip file to directory."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        return True
    except Exception:
        return False


def _count_samples_in_files(annotation_files: List[str], dataset_dir: str) -> int:
    """Count total samples across annotation files."""
    total_samples = 0
    for f in annotation_files:
        file_path = os.path.join(dataset_dir, "bddoia-fiftyone-v2", f)
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as file:
                    data = json.load(file)
                    total_samples += len(data.get("images", []))
            except Exception:
                continue
    return total_samples


def _process_category_vector(category_vec: Any, category_mapping: Dict[int, str]) -> List[str]:
    """Process category vector and return list of action names (multilabel)."""
    labels = []
    if isinstance(category_vec, list):
        for i, val in enumerate(category_vec):
            if val == 1 and i in category_mapping and category_mapping[i] != "confuse":
                labels.append(category_mapping[i])
    elif isinstance(category_vec, int):
        if category_vec in category_mapping and category_mapping[category_vec] != "confuse":
            labels.append(category_mapping[category_vec])
    return labels


def _process_reason_vector(reason_vec: List[int]) -> List[str]:
    """Process reason vector and return reason labels."""
    if not isinstance(reason_vec, list):
        return []
    
    return [
        REASON_CLASSES[i] for i, val in enumerate(reason_vec)
        if val == 1 and i < len(REASON_CLASSES)
    ]


def _process_objects(frame_objects: List[Dict[str, Any]], image_info: Dict[str, Any]) -> Tuple[List[fo.Detection], List[fo.Polyline]]:
    """Process frame objects and return detections and polylines."""
    detections, polylines = [], []

    w, h = float(image_info.get("height")), float(image_info.get("width"))
    
    for obj in frame_objects:
        if "box2d" in obj and "category" in obj:
            x1, y1, x2, y2 = (obj["box2d"][k] for k in ("x1", "y1", "x2", "y2"))
            norm = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]
            
            detection = fo.Detection(
                label=obj["category"],
                bounding_box=norm,
            )
            
            if "attributes" in obj:
                attrs = obj["attributes"]
                if "occluded" in attrs:
                    detection["occluded"] = attrs["occluded"]
                if "truncated" in attrs:
                    detection["truncated"] = attrs["truncated"]
                if "trafficLightColor" in attrs:
                    detection["traffic_light_color"] = attrs["trafficLightColor"]
            
            detections.append(detection)
        
        elif "poly2d" in obj and "category" in obj:            
            formatted_points = [[
                (p[0] / w, p[1] / h) 
                for p in obj["poly2d"] 
                if isinstance(p, (list, tuple)) and len(p) >= 2
            ]]
            
            polyline = fo.Polyline(
                label=obj["category"],
                points=formatted_points,
                closed=False,
            )
            polylines.append(polyline)
    
    return detections, polylines


def _create_sample_data(ann: Dict[str, Any], image_info: Dict[str, Any], 
                       dataset_dir: str, split_name: str, 
                       category_mapping: Dict[int, str]) -> Optional[Dict[str, Any]]:
    """Create sample data from annotation."""
    filepath = os.path.join(dataset_dir, "bddoia-fiftyone-v2", "data", image_info["file_name"])
    
    if not os.path.exists(filepath):
        return None
    
    sample_data = {
        "filepath": filepath,
        "split": split_name,
        "image_id": ann.get("id", "")
    }
    
    sample_data["ground_truth"] = fo.Classifications(
        classifications=[fo.Classification(label=label) for label in _process_category_vector(ann.get("category", []), category_mapping)]
    )
    sample_data["unsafe_action"] = fo.Classifications(
        classifications=[fo.Classification(label=label) for label in _process_category_vector(ann.get("unsafe", []), category_mapping)]
    )
    sample_data["reasons"] = _process_reason_vector(ann.get("reason", "unknown"))    

    attrs = ann.get("attributes", {})
    sample_data["weather"] = fo.Classification(label=attrs.get("weather", "clear"))
    sample_data["scene"] = fo.Classification(label=attrs.get("scene", "city street"))
    sample_data["timeofday"] = fo.Classification(label=attrs.get("timeofday", "daytime"))    
    return sample_data

def download_and_prepare(dataset_dir: str, split: Optional[str] = None, 
                        classes: Optional[List[str]] = None, 
                        max_samples: Optional[int] = None, **kwargs) -> Tuple[Optional[str], int, List[str]]:
    """Download and prepare the dataset."""
    os.makedirs(dataset_dir, exist_ok=True)
    
    if split is not None and split not in SPLIT_MAPPING:
        raise ValueError(f"Invalid split '{split}'. Supported splits: {list(SPLIT_MAPPING.keys())}")
    
    annotation_files = [SPLIT_MAPPING[split]] if split else list(SPLIT_MAPPING.values())
    images_dir = os.path.join(dataset_dir, "bddoia-fiftyone-v2", "data")
    
    if all(os.path.exists(os.path.join(dataset_dir, "bddoia-fiftyone-v2", f)) for f in annotation_files) and os.path.exists(images_dir):
        total_samples = _count_samples_in_files(annotation_files, dataset_dir)
        return None, total_samples, ACTION_CLASSES
    
    dataset_url = "https://cdn.voxel51.com/datasets/bddoia-fiftyone-v2.zip"
    zip_path = os.path.join(dataset_dir, "bddoia-fiftyone-v2.zip")
    
    if not os.path.exists(zip_path):
        if not _download_file(dataset_url, zip_path):
            return None, 0, []
    
    if not _extract_zip(zip_path, dataset_dir):
        return None, 0, []
    
    total_samples = _count_samples_in_files(annotation_files, dataset_dir)
    
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    return None, total_samples, ACTION_CLASSES


def load_dataset(dataset: fo.Dataset, dataset_dir: str, split: Optional[str] = None,
                classes: Optional[List[str]] = None, max_samples: Optional[int] = None,
                shuffle: bool = False, seed: Optional[int] = None, **kwargs) -> fo.Dataset:
    """Load the dataset into FiftyOne."""
    if split is not None and split not in SPLIT_MAPPING:
        raise ValueError(f"Invalid split '{split}'. Supported splits: {list(SPLIT_MAPPING.keys())}")
    
    splits_to_load = [split] if split else list(SPLIT_MAPPING.keys())
    
    action_classes = ACTION_CLASSES.copy()
    if classes is not None:
        classes = [classes] if isinstance(classes, str) else classes
        invalid_classes = [c for c in classes if c not in action_classes]
        if invalid_classes:
            raise ValueError(f"Invalid classes: {invalid_classes}. Available classes: {action_classes}")
        action_classes = classes
    
    if seed is not None:
        random.seed(seed)
    
    total_loaded = 0
    for split_name in splits_to_load:
        annotation_file = SPLIT_MAPPING[split_name]
        annotation_path = os.path.join(dataset_dir, "bddoia-fiftyone-v2", annotation_file)
        
        if not os.path.exists(annotation_path):
            continue
            
        try:
            with open(annotation_path, 'r') as f:
                data = json.load(f)
        except Exception:
            continue
        
        image_lookup = {img["id"]: img for img in data.get("images", [])}
        
        categories = data.get("categories", [])
        category_mapping = (
            {cat['category_id']: cat['name'] for cat in categories}
            if categories else {i: name for i, name in enumerate(ACTION_CLASSES)}
        )
        
        annotations = data.get("annotations", [])
        
        if classes is not None:
            annotations = [
                ann for ann in annotations
                if "category" in ann and any(label in classes for label in _process_category_vector(ann["category"], category_mapping))
            ]
        
        if max_samples is not None and len(annotations) > max_samples:
            if shuffle:
                import random
                random.shuffle(annotations)
            annotations = annotations[:max_samples]
        
        samples_data = []
        
        for ann in annotations:
            image_info = image_lookup.get(ann["id"])
            if not image_info:
                continue
            
            sample_data = _create_sample_data(ann, image_info, dataset_dir, split_name, category_mapping)
            if sample_data:
                samples_data.append(sample_data)
        
        if samples_data:
            samples = [fo.Sample(**sample_data) for sample_data in samples_data]
            dataset.add_samples(samples)
            total_loaded += len(samples)
    
    return dataset
    