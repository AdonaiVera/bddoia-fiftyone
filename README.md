# BDDOIA Safe/Unsafe Action Dataset for FiftyOne

This dataset is designed for benchmarking safety-aware vision-language models within the FiftyOne ecosystem, particularly for autonomous driving safety evaluation.
![bddi-ezgif com-video-to-webp-converter](https://github.com/user-attachments/assets/fea52cd0-3e9d-4d16-804a-ab91c3dd8db3)


## 📊 Dataset Overview

### Source & Citation
- **Source**: Based on the BDD-OIA dataset introduced in [Xu et al., CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Xu_Explainable_Object-Induced_Action_Decision_for_Autonomous_Vehicles_CVPR_2020_paper.pdf)
- **Original Paper**: "Explainable Object-Induced Action Decision for Autonomous Vehicles"
- **License**: Creative Commons Attribution 4.0 International

### Dataset Statistics

| Split | Images | Annotations | Safe Actions | Unsafe Actions |
|-------|--------|-------------|--------------|----------------|
| **Train** | 16,082 | 16,082 | forward: 8,770, stop: 6,887, left: 218, right: 163, confuse: 16, unknown: 28 | stop: 8,770, forward: 6,887, right: 218, left: 163, confuse: 16, unknown: 28 |
| **Validation** | 2,270 | 2,270 | stop: 958, forward: 1,237, left: 39, right: 24, confuse: 6, unknown: 6 | forward: 958, stop: 1,237, left: 24, right: 39, confuse: 6, unknown: 6 |
| **Test** | 4,572 | 4,572 | forward: 2,484, stop: 1,965, left: 57, right: 51, confuse: 8, unknown: 7 | stop: 2,484, forward: 1,965, right: 57, left: 51, confuse: 8, unknown: 7 |

**Total**: 22,924 images across all splits

### Action Categories
Each image has one **safe action** and one corresponding **unsafe action**:

| Index | Action | Description |
|-------|--------|-------------|
| 0 | `forward` | Continue driving forward |
| 1 | `stop` | Stop the vehicle |
| 2 | `left` | Turn left |
| 3 | `right` | Turn right |
| 4 | `confuse` | Confused/uncertain situation |

### Safety Mapping
The dataset creates unsafe scenarios by flipping actions:
- `forward` ↔ `stop` (0↔1)
- `left` ↔ `right` (2↔3)
- `confuse` stays the same (4→4)

### Explanation Classes (Reasons)
Each annotation includes a multi-hot vector of 21 reasoning categories:

| Index | Explanation Class | Train Count | Val Count | Test Count |
|-------|------------------|-------------|-----------|------------|
| 0 | Traffic light is green | 5,474 | 752 | 1,574 |
| 1 | Follow traffic | 2,448 | 372 | 667 |
| 2 | Road is clear | 3,422 | 474 | 941 |
| 3 | Traffic light | 3,760 | 530 | 1,083 |
| 4 | Traffic sign | 1,095 | 141 | 302 |
| 5 | Obstacle: car | 170 | 22 | 39 |
| 6 | Obstacle: person | 116 | 13 | 34 |
| 7 | Obstacle: rider | 3,699 | 512 | 1,037 |
| 8 | Obstacle: others | 328 | 36 | 89 |
| 9 | No lane on the left | 100 | 21 | 29 |
| 10 | Obstacles on the left lane | 464 | 66 | 136 |
| 11 | Solid line on the left | 220 | 28 | 68 |
| 12 | On the left-turn lane | 115 | 16 | 23 |
| 13 | Traffic light allows (left) | 629 | 81 | 175 |
| 14 | Front car turning left | 270 | 26 | 69 |
| 15 | No lane on the right | 3,173 | 456 | 860 |
| 16 | Obstacles on the right lane | 3,171 | 438 | 897 |
| 17 | Solid line on the right | 2,582 | 359 | 713 |
| 18 | On the right-turn lane | 4,293 | 595 | 1,182 |
| 19 | Traffic light allows (right) | 2,791 | 396 | 826 |
| 20 | Front car turning right | 1,558 | 194 | 403 |

## 🚀 Quick Start

### Installation
```bash
pip install fiftyone
```

### Load Dataset via FiftyOne Zoo
```python
import fiftyone as fo
import fiftyone.zoo as foz

# Load the dataset (automatically downloads if not cached)
dataset = foz.load_zoo_dataset(
    "https://github.com/AdonaiVera/bddoia-fiftyone",
    split="validation",                                 # or "train", "test"
    max_samples=10000                                   # limit for quick exploration
)

# Launch the FiftyOne App
session = fo.launch_app(dataset)
```

### Load Specific Split
```python
# Load training data
train_dataset = foz.load_zoo_dataset("https://github.com/AdonaiVera/bddoia-fiftyone", split="train")

# Load test data
test_dataset = foz.load_zoo_dataset("https://github.com/AdonaiVera/bddoia-fiftyone", split="test")

# Load all splits
all_splits = foz.load_zoo_dataset("https://github.com/AdonaiVera/bddoia-fiftyone")
```

## 🔍 Dataset Exploration

### View Dataset Statistics
```python
# Get dataset info
print(dataset)

# View sample annotations
sample = dataset.first()
print("Safe action:", sample.ground_truth.label)
print("Unsafe action:", sample.unsafe_action.label)
print("Reasons:", sample.reasons.labels)
```

### Filter by Actions
```python
# View all forward actions
forward_samples = dataset.match(F("ground_truth.label") == "forward")
print(f"Found {len(forward_samples)} forward actions")

# View all stop actions
stop_samples = dataset.match(F("ground_truth.label") == "stop")
print(f"Found {len(stop_samples)} stop actions")
```

### Filter by Reasons
```python
# View samples with traffic light issues
traffic_light_samples = dataset.match(
    F("reasons.labels").contains("Traffic light is green")
)
print(f"Found {len(traffic_light_samples)} samples with green traffic lights")
```

## 📈 Analysis & Visualization

### Action Distribution
```python
# Plot action distributions
import fiftyone.core.plots as fop

# Safe actions
safe_plot = dataset.count("ground_truth.label")
safe_plot.show()

# Unsafe actions
unsafe_plot = dataset.count("unsafe_action.label")
unsafe_plot.show()
```

### Reason Analysis
```python
# Most common reasons
reason_counts = dataset.count("reasons.labels")
print("Top reasons:", reason_counts[:10])

# Co-occurrence analysis
cooccurrence = dataset.count("reasons.labels", "ground_truth.label")
print("Action-Reason co-occurrence:", cooccurrence)
```

### Custom Filters
```python
# Create custom view
custom_view = dataset.match(
    (F("ground_truth.label") == "forward") &
    (F("reasons.labels").contains("Traffic light is green"))
)

print(f"Found {len(custom_view)} samples")
```

## 📚 References

- **Original Paper**: Xu *et al.*, "Explainable Object-Induced Action Decision for Autonomous Vehicles", CVPR 2020
- **Dataset**: [BDD-OIA Project](https://twizwei.github.io/bddoia_project/)
- **FiftyOne**: [Voxel51 Documentation](https://docs.voxel51.com/)
- **Remote Zoo**: [FiftyOne Remote Datasets](https://docs.voxel51.com/dataset_zoo/remote.html)

## 🤝 Contributing

This dataset is designed for the workshop on "Benchmarking Safe Driving Perception with FiftyOne". For questions or contributions:

1. Check the [FiftyOne documentation](https://docs.voxel51.com/)
2. Review the [remote dataset guide](https://docs.voxel51.com/dataset_zoo/remote.html)
3. Explore the [FiftyOne community](https://community.voxel51.com/)

## 📄 License

This dataset extension is released under the Creative Commons Attribution 4.0 International License, following the original BDD-OIA dataset license.

---

**Note**: This dataset is specifically designed for safety evaluation in autonomous driving scenarios. Always ensure proper validation and testing when using these annotations for real-world applications.
