from pathlib import Path
import xml.etree.ElementTree as ET
import shutil
import random

# Original dataset root
NEU_DET_ROOT = Path("NEU-DET")

# New root for YOLO-style dataset
yolo_dataset_name = "dataset_yolo"
YOLO_ROOT = Path(yolo_dataset_name)

# Class mapping: <name> in XML -> YOLO class id
CLASS_NAME_TO_ID = {
    "crazing": 0,
    "inclusion": 1,
    "patches": 2,
    "pitted_surface": 3,
    "rolled-in_scale": 4, 
    "scratches": 5,
}

# Class names in order of their IDs (must match mapping above)
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

# Split ratios (75% train, 15% val, 10% test)
TRAIN_RATIO = 0.75
VAL_RATIO = 0.15  

RANDOM_SEED = 42 # for reproducibility   

# Directory creation
def make_yolo_dirs():
    """
    Create the target structure:
        images/train/images, images/train/labels, ...
    """
    for split in ["train", "val", "test"]:
        (YOLO_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (YOLO_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)


# Collect all XML files from train/ and validation/ annotations
def collect_all_xmls():
    """
    Collect all XML annotation files from:
        NEU-DET/train/annotations
        NEU-DET/validation/annotations
    """
    xml_paths = []
    for subset in ["train", "validation"]:
        ann_dir = NEU_DET_ROOT / subset / "annotations"
        xml_paths.extend(sorted(ann_dir.glob("*.xml")))
    return xml_paths

# Find image file corresponding to given filename from XML
def find_image_file(filename: str) -> Path:
    """
    Find the corresponding image file for a given filename from XML.
    Some XMLs may have <filename> without extension (e.g. 'patches_123'),
    so if no extension is present we try adding '.jpg' and '.JPG'.

    Search is done under:
        NEU-DET/train/images/**/
        NEU-DET/validation/images/**/
    """
    img_roots = [
        NEU_DET_ROOT / "train" / "images",
        NEU_DET_ROOT / "validation" / "images",
    ]

    # If filename has no dot, assume it's missing the extension → add .jpg/.JPG
    if "." in filename:
        candidate_names = [filename]
    else:
        candidate_names = [filename + ".jpg", filename + ".JPG"]

    # Search both roots recursively
    for root in img_roots:
        for name in candidate_names:
            # rglob searches all subdirectories for this file name
            for path in root.rglob(name):
                if path.is_file():
                    return path

    raise FileNotFoundError(
        f"Could not find image file for '{filename}' (tried: {candidate_names})."
    )

# VOC to YOLO conversion
def voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    """
    Convert VOC box (absolute pixels) -> YOLO box (normalized).
    """
    x_c = (xmin + xmax) / 2.0
    y_c = (ymin + ymax) / 2.0
    bw = xmax - xmin
    bh = ymax - ymin

    # normalize to [0,1]
    x_c /= img_w
    y_c /= img_h
    bw /= img_w
    bh /= img_h

    # clamp for safety
    x_c = min(max(x_c, 0.0), 1.0)
    y_c = min(max(y_c, 0.0), 1.0)
    bw = min(max(bw, 0.0), 1.0)
    bh = min(max(bh, 0.0), 1.0)

    return x_c, y_c, bw, bh

# XML parsing
def parse_xml(xml_path: Path):
    """
    Parse one XML and return:
      - image filename (as in <filename>)
      - image width, height
      - list of objects with (class_id, xmin, ymin, xmax, ymax)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename_tag = root.find("filename")
    if filename_tag is None or not filename_tag.text:
        raise ValueError(f"{xml_path.name} has no <filename>")

    img_filename = filename_tag.text.strip()

    size_tag = root.find("size")
    if size_tag is None:
        raise ValueError(f"{xml_path.name} has no <size>")

    img_w = int(size_tag.find("width").text)
    img_h = int(size_tag.find("height").text)

    objects = []
    for obj in root.findall("object"):
        name_tag = obj.find("name")
        if name_tag is None or not name_tag.text:
            continue

        class_name = name_tag.text.strip()
        if class_name not in CLASS_NAME_TO_ID:
            print(f"[WARN] Unknown class '{class_name}' in {xml_path.name}; skipping object")
            continue

        class_id = CLASS_NAME_TO_ID[class_name]

        bnd = obj.find("bndbox")
        if bnd is None:
            continue

        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)

        objects.append((class_id, xmin, ymin, xmax, ymax))

    return img_filename, img_w, img_h, objects

# Write YOLO label file
def write_yolo_label(label_path: Path, img_w: int, img_h: int, objects):
    """
    Write YOLO label file for a single image.
    One line per object:
        class_id x_center y_center width height
    """
    lines = []
    for class_id, xmin, ymin, xmax, ymax in objects:
        x_c, y_c, bw, bh = voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
        lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

    with open(label_path, "w") as f:
        for line in lines:
            f.write(line + "\n")

# Create dataset YAML file
def create_yaml():
    """
    Create neu_det.yaml in NEU-DET root, pointing to the new structure.
    YOLO will look for labels automatically under the matching 'labels' dirs.
    """
    yaml_path = YOLO_ROOT / "neu_det.yaml"

    yaml_text = f"""# NEU-DET dataset config for YOLO11
path: {yolo_dataset_name}

train: train/images
val: val/images
test: test/images

nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""

    with open(yaml_path, "w") as f:
        f.write(yaml_text)

    print(f"[INFO] Wrote dataset YAML to {yaml_path}")

# Main processing
if __name__ == "__main__":
    print(f"[INFO] NEU-DET root: {NEU_DET_ROOT.resolve()}")

    # Create target directories
    make_yolo_dirs()

    # Collect and shuffle all XMLs (from original train + validation)
    xml_files = collect_all_xmls()
    if not xml_files:
        raise RuntimeError("No XML files found in train/ or validation/ annotations.")

    random.seed(RANDOM_SEED)
    random.shuffle(xml_files)

    n_total = len(xml_files)
    n_train = int(TRAIN_RATIO * n_total)
    n_val = int(VAL_RATIO * n_total)
    n_test = n_total - n_train - n_val

    train_xmls = xml_files[:n_train]
    val_xmls = xml_files[n_train:n_train + n_val]
    test_xmls = xml_files[n_train + n_val:]

    print(f"[INFO] Total samples: {n_total}")
    print(f"[INFO] Split -> train: {len(train_xmls)}, val: {len(val_xmls)}, test: {len(test_xmls)}")

    # Map each XML to its split name
    split_map = {}
    for x in train_xmls:
        split_map[x] = "train"
    for x in val_xmls:
        split_map[x] = "val"
    for x in test_xmls:
        split_map[x] = "test"

    # Process each XML: parse, locate image, copy, write label
    for xml_path in xml_files:
        split = split_map[xml_path]

        try:
            img_filename, img_w, img_h, objects = parse_xml(xml_path)
        except Exception as e:
            print(f"[WARN] Failed to parse {xml_path.name}: {e}")
            continue

        try:
            src_img = find_image_file(img_filename)
        except FileNotFoundError as e:
            print(f"[WARN] {e} Skipping this sample.")
            continue

        dst_img_dir = YOLO_ROOT / split / "images"
        dst_lbl_dir = YOLO_ROOT / split / "labels"

        dst_img = dst_img_dir / (src_img.name)  # keep original filename + extension
        dst_lbl = dst_lbl_dir / (src_img.stem + ".txt")

        # copy image
        shutil.copy2(src_img, dst_img)

        # write label file
        write_yolo_label(dst_lbl, img_w, img_h, objects)

    # Create dataset YAML
    create_yaml()

    print(f"[INFO] Done. YOLO-style dataset created under '{yolo_dataset_name}/'.")
