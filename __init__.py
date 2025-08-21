"""
BDDOIA Safe/Unsafe Action Dataset for FiftyOne

A comprehensive COCO-format extension of the BDD-OIA dataset with explicit 
unsafe action labels and reason explanations. Designed for benchmarking 
safety-aware vision-language models in autonomous driving scenarios.
"""

import os
import json
import zipfile
import requests
import fiftyone as fo

SPLIT_MAPPING = {
    "train": "train_25k_images.json",
    "validation": "val_25k_images.json", 
    "test": "test_25k_images.json"
}

ACTION_CLASSES = ["forward", "stop", "left", "right", "confuse"]

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


def download_and_prepare(dataset_dir, split=None, classes=None, max_samples=None, **kwargs):
    os.makedirs(dataset_dir, exist_ok=True)
    
    if split is not None and split not in SPLIT_MAPPING:
        raise ValueError(f"Invalid split '{split}'. Supported splits: {list(SPLIT_MAPPING.keys())}")
    
    annotation_files = list(SPLIT_MAPPING.values())
    images_dir = os.path.join(dataset_dir, "data")
    
    if split is not None:
        annotation_files = [SPLIT_MAPPING[split]]
    
    if all(os.path.exists(os.path.join(dataset_dir, f)) for f in annotation_files) and os.path.exists(images_dir):
        total_samples = 0
        for f in annotation_files:
            if os.path.exists(os.path.join(dataset_dir, f)):
                try:
                    with open(os.path.join(dataset_dir, f), 'r') as file:
                        data = json.load(file)
                        total_samples += len(data.get("images", []))
                except Exception:
                    pass
        return None, total_samples, ACTION_CLASSES
    
    dataset_url = "https://cdn.voxel51.com/datasets/bddoia-fiftyone-v1.zip"
    zip_path = os.path.join(dataset_dir, "bddoia-fiftyone-v1.zip")
    
    if not os.path.exists(zip_path):
        try:
            response = requests.get(dataset_url, stream=True)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None, 0, []
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_dir)
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        return None, 0, []
    
    total_samples = 0
    for f in annotation_files:
        if os.path.exists(os.path.join(dataset_dir, f)):
            try:
                with open(os.path.join(dataset_dir, f), 'r') as file:
                    data = json.load(file)
                    total_samples += len(data.get("images", []))
            except Exception:
                pass
    
    if os.path.exists(zip_path):
        os.remove(zip_path)
    
    return None, total_samples, ACTION_CLASSES


def load_dataset(dataset, dataset_dir, split=None, classes=None, max_samples=None, shuffle=False, seed=None, **kwargs):
    if split is not None and split not in SPLIT_MAPPING:
        raise ValueError(f"Invalid split '{split}'. Supported splits: {list(SPLIT_MAPPING.keys())}")
    
    if split is None:
        splits_to_load = list(SPLIT_MAPPING.keys())
    else:
        splits_to_load = [split]
    
    action_classes = ACTION_CLASSES.copy()
    if classes is not None:
        if isinstance(classes, str):
            classes = [classes]
        invalid_classes = [c for c in classes if c not in action_classes]
        if invalid_classes:
            raise ValueError(f"Invalid classes: {invalid_classes}. Available classes: {action_classes}")
        action_classes = classes
    
    if seed is not None:
        import random
        random.seed(seed)
    
    total_loaded = 0
    for split_name in splits_to_load:
        annotation_file = SPLIT_MAPPING[split_name]
        annotation_path = os.path.join(dataset_dir, annotation_file)
        
        if not os.path.exists(annotation_path):
            continue
            
        try:
            with open(annotation_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            continue
        
        image_lookup = {img["id"]: img for img in data.get("images", [])}
        
        categories = data.get("categories", [])
        if categories:
            category_mapping = {cat['category_id']: cat['name'] for cat in categories}
        else:
            category_mapping = {i: name for i, name in enumerate(ACTION_CLASSES)}
        
        annotations = data.get("annotations", [])
        
        if classes is not None:
            annotations = [
                ann for ann in annotations
                if "category" in ann and (
                    (isinstance(ann["category"], list) and 1 in ann["category"] and 
                     ann["category"].index(1) < len(category_mapping) and 
                     category_mapping[ann["category"].index(1)] in classes) or
                    (isinstance(ann["category"], int) and 
                     ann["category"] < len(category_mapping) and 
                     category_mapping[ann["category"]] in classes)
                )
            ]
        
        if max_samples is not None and len(annotations) > max_samples:
            if shuffle:
                random.shuffle(annotations)
            annotations = annotations[:max_samples]
        
        samples_data = []
        
        for ann in annotations:
            image_info = image_lookup.get(ann["id"])
            if not image_info:
                continue
                
            filepath = os.path.join(dataset_dir, "data", image_info["file_name"])
            
            if not os.path.exists(filepath):
                continue
            
            sample_data = {
                "filepath": filepath,
                "split": split_name,
                "image_id": ann.get("id", "")
            }
            
            if "category" in ann:
                category_vec = ann["category"]
                if isinstance(category_vec, list) and 1 in category_vec:
                    safe_idx = category_vec.index(1)
                    if safe_idx in category_mapping:
                        safe_action_name = category_mapping[safe_idx]
                        sample_data["ground_truth"] = fo.Classification(label=safe_action_name)
                elif isinstance(category_vec, int) and category_vec in category_mapping:
                    safe_action_name = category_mapping[category_vec]
                    sample_data["ground_truth"] = fo.Classification(label=safe_action_name)
            
            if "unsafe" in ann:
                unsafe_vec = ann["unsafe"]
                if isinstance(unsafe_vec, list) and 1 in unsafe_vec:
                    unsafe_idx = unsafe_vec.index(1)
                    if unsafe_idx in category_mapping:
                        unsafe_action_name = category_mapping[unsafe_idx]
                        sample_data["unsafe_action"] = fo.Classification(label=unsafe_action_name)
                elif isinstance(unsafe_vec, int) and unsafe_vec in category_mapping:
                    unsafe_action_name = category_mapping[unsafe_vec]
                    sample_data["unsafe_action"] = fo.Classification(label=unsafe_action_name)
            
            if "reason" in ann:
                reason_vec = ann["reason"]
                if isinstance(reason_vec, list):
                    reasons = [
                        REASON_CLASSES[i] for i, val in enumerate(reason_vec)
                        if val == 1 and i < len(REASON_CLASSES)
                    ]
                    if reasons:
                        sample_data["reasons"] = fo.Classifications(labels=reasons)
                    else:
                        sample_data["reasons"] = fo.Classifications(labels=[])
            
            if "frames" in ann and ann["frames"]:
                for frame in ann["frames"]:
                    if "objects" in frame:
                        detections = []
                        polylines = []
                        
                        for obj in frame["objects"]:
                            if "box2d" in obj and "category" in obj:
                                bbox = obj["box2d"]
                                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                                
                                detection = fo.Detection(
                                    label=obj["category"],
                                    bounding_box=[x1, y1, x2 - x1, y2 - y1],
                                    confidence=1.0
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
                                polyline = fo.Polyline(
                                    label=obj["category"],
                                    points=obj["poly2d"],
                                    confidence=1.0
                                )
                                
                                if "attributes" in obj:
                                    attrs = obj["attributes"]
                                    if "direction" in attrs:
                                        polyline["direction"] = attrs["direction"]
                                    if "style" in attrs:
                                        polyline["style"] = attrs["style"]
                                
                                polylines.append(polyline)
                        
                        if detections:
                            sample_data["detections"] = fo.Detections(detections=detections)
                        if polylines:
                            sample_data["polylines"] = fo.Polylines(polylines=polylines)
            
            if "attributes" in ann:
                attrs = ann["attributes"]
                if "weather" in attrs and attrs["weather"] != "undefined":
                    sample_data["weather"] = fo.Classification(label=attrs["weather"])
                if "scene" in attrs:
                    sample_data["scene"] = fo.Classification(label=attrs["scene"])
                if "timeofday" in attrs:
                    sample_data["timeofday"] = fo.Classification(label=attrs["timeofday"])
            
            if "ground_truth" not in sample_data and "unsafe_action" not in sample_data:
                continue
            
            for key, value in ann.items():
                if key not in ["id", "category", "unsafe", "reason"] and key not in sample_data:
                    sample_data[f"annotation_{key}"] = value
            
            samples_data.append(sample_data)
        
        if samples_data:
            samples = [fo.Sample(**sample_data) for sample_data in samples_data]
            dataset.add_samples(samples)
            total_loaded += len(samples)
    
    return dataset
    