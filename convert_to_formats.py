import json
import os

# Paths
input_file = "annotations/via_project.json"
yolo_dir = "annotations/yolo"
coco_file = "annotations/coco_annotations.json"

# Assumed image dimensions (replace with actual dimensions if known)
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Load VIA data
with open(input_file, "r") as f:
    via_data = json.load(f)

label_map = {"cat": 0, "dog": 1}
yolo_data = {}
coco = {
    "images": [],
    "annotations": [],
    "categories": [{"id": 0, "name": "cat"}, {"id": 1, "name": "dog"}]
}
annotation_id = 1
image_id = 1

# Create YOLO directory if it doesn't exist
os.makedirs(yolo_dir, exist_ok=True)

for file_id, entry in via_data.items():
    filename = entry["filename"]
    regions = entry["regions"]

    # YOLO .txt content
    yolo_lines = []
    for region in regions:
        shape = region["shape_attributes"]
        label = region["region_attributes"].get("label", "cat")
        class_id = label_map[label]

        x_center = (shape["x"] + shape["width"] / 2) / IMAGE_WIDTH
        y_center = (shape["y"] + shape["height"] / 2) / IMAGE_HEIGHT
        width = shape["width"] / IMAGE_WIDTH
        height = shape["height"] / IMAGE_HEIGHT

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # COCO annotation
        coco["annotations"].append({
            "id": annotation_id,
            "image_id": image_id,
            "category_id": class_id,
            "bbox": [shape["x"], shape["y"], shape["width"], shape["height"]],
            "area": shape["width"] * shape["height"],
            "iscrowd": 0
        })
        annotation_id += 1

    # Save YOLO file
    if yolo_lines:
        yolo_filename = os.path.splitext(filename)[0] + ".txt"
        with open(os.path.join(yolo_dir, yolo_filename), "w") as f:
            f.write("\n".join(yolo_lines))

    # Add COCO image info
    coco["images"].append({
        "id": image_id,
        "file_name": filename,
        "width": IMAGE_WIDTH,
        "height": IMAGE_HEIGHT
    })

    image_id += 1

# Save COCO file
with open(coco_file, "w") as f:
    json.dump(coco, f, indent=2)

print("✅ Conversion complete!")
