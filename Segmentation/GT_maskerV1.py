import pydicom
import csv
import json
from pathlib import Path
from zipfile import ZipFile, BadZipFile
import itertools
import numpy as np
import cv2
import sys

    # ==== 1. change my absolute path ====
ROOT_DIR = Path("D:/Uni work/Year 3/Thrombus+/Internal (Post June)/thrombus_B1_dataset")    #removed LT1 as problematic

CSV_COLUMNS = [
    "dicom_file", "zip_path", "json_member",
    "width", "height", "frame_index", "plane", "series_instance_uid",
    "label", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "polygon_paths", "acep_grade",
]

# \/\/\/ File Readers \/\/\/ #
def get_json_path(DIRECTORY):
    '''Collects JSON file directorys within ZIP files of DIR'''
    json_paths = []
    for zip_path in DIRECTORY.rglob("*.zip"):
        try:
            with ZipFile(zip_path, 'r') as open_zip_ref:
                for member in open_zip_ref.namelist():
                    if member.lower().endswith(".json"):
                        json_paths.append((member, zip_path))
        except:
            print(f"Skipped erroneous zip: {zip_path}")
    return json_paths

#print(f"{get_json_path(ROOT_DIR)}")

def read_json(json_paths):
    '''Reads width, height, coordinate points, frame index, Location/label, ACEP grade'''
    rows = []

    for member, zip_path in json_paths:
        try:
            with ZipFile(zip_path, "r") as archive:
                with archive.open(member) as json_file:
                    data = json.load(json_file)
        except (BadZipFile, KeyError, OSError, json.JSONDecodeError) as error:
            print(f"Skipped {zip_path} :: {member}: {error}")
            continue

        item = data.get("item", {})
        slots = {
            slot.get("slot_name"): slot
            for slot in item.get("slots", [])
        }
        annotations = data.get("annotations", [])

        # fetch ACEP grade
        grades_by_frame = {}
        for annotation in annotations:
            for prop in annotation.get("properties", []):
                if prop.get("name") == "ACEP Grading Score":
                    frame = str(prop.get("frame_index", ""))
                    grades_by_frame[frame] = prop.get("value", "")

        for annotation in annotations:
            slot_names = annotation.get("slot_names", [])
            if slot_names:
                slot = slots.get(slot_names[0], {})
            else:
                slot = next(iter(slots.values()), {})

            metadata = slot.get("metadata", {})

            for frame_index, frame in annotation.get("frames", {}).items():
                polygon = frame.get("polygon")
                if not polygon:
                    continue

                bbox = frame.get("bounding_box", {})
                frame_index = str(frame_index)

                rows.append({
                    "dicom_file": item.get("name", ""),
                    "zip_path": str(zip_path),
                    "json_member": member,
                    "width": slot.get("width", ""),
                    "height": slot.get("height", ""),
                    "frame_index": frame_index,
                    "plane": metadata.get("plane_map", {}).get(
                        frame_index, metadata.get("primary_plane", "")
                    ),
                    "series_instance_uid": metadata.get("SeriesInstanceUID", ""),
                    "label": annotation.get("name", ""),
                    "bbox_x": bbox.get("x", ""),
                    "bbox_y": bbox.get("y", ""),
                    "bbox_w": bbox.get("w", ""),
                    "bbox_h": bbox.get("h", ""),
                    "polygon_paths": json.dumps(polygon.get("paths", [])),
                    "acep_grade": grades_by_frame.get(
                        frame_index, grades_by_frame.get("", "")
                    ),
                })

    return rows

# \/\/\/ csv and mask constructors \/\/\/
def write_to_csv(CSV_DIR, rows):
    with CSV_DIR.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

def set_csv_field_size_limit():
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10

def create_gt_masks(csv_path, mask_dir):
    '''collates all possible location labels, reads point coordinates for each frame, draws GT masks as and saves as pngs'''
    set_csv_field_size_limit()
    # First pass collects labels
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        labels = sorted({
            row["label"]
            for row in csv.DictReader(csv_file)
            if row["label"]
        })

    label_ids = {label: index for index, label in enumerate(labels, start=1)}
    mask_dir.mkdir(parents=True, exist_ok=True)

    with (mask_dir / "label_map.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["label_id", "label"])
        writer.writerows((label_id, label) for label, label_id in label_ids.items())

    masks_written = 0
    # Second pass processes one rows at a time
    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        key = lambda row: (row["zip_path"], row["json_member"], row["dicom_file"])

        for (zip_path, _json_member, dicom_file), grouped_rows in itertools.groupby(
            reader, key=key
        ):
            rows = list(grouped_rows)

            try:
                with ZipFile(zip_path) as archive:
                    with archive.open(dicom_file) as stream:
                        dicom = pydicom.dcmread(stream, stop_before_pixels=True)

                    height, width = int(dicom.Rows), int(dicom.Columns)
                    frame_count = int(getattr(dicom, "NumberOfFrames", 1))
                    source_width = float(rows[0]["width"] or width)
                    source_height = float(rows[0]["height"] or height)

                    frame_masks = {}

                    for row in rows:
                        frame_index = int(row["frame_index"])
                        if not 0 <= frame_index < frame_count:
                            print(f"Skipping invalid frame {frame_index}: {dicom_file}")
                            continue

                        mask = frame_masks.setdefault(
                            frame_index,
                            np.zeros((height, width), dtype=np.uint8),
                        )

                        paths = json.loads(row["polygon_paths"] or "[]")
                        for path in paths:
                            points = np.asarray([[point["x"], point["y"]] if isinstance(point, dict) else point for point in path],
                                                dtype=np.float32,)
                            
                            if len(points) < 3:
                                continue

                            points[:, 0] *= width / source_width
                            points[:, 1] *= height / source_height
                            points = np.rint(points).astype(np.int32).reshape((-1, 1, 2))
                            cv2.fillPoly(mask, [points], label_ids[row["label"]])

                relative_zip = Path(zip_path).relative_to(ROOT_DIR)
                output_dir = mask_dir / relative_zip.parent / relative_zip.stem
                output_dir.mkdir(parents=True, exist_ok=True)

                for frame_index, mask in frame_masks.items():
                    filename = f"{Path(dicom_file).stem}_frame_{frame_index:04d}.png"
                    if not cv2.imwrite(str(output_dir / filename), mask):
                        raise OSError(f"Could not write {output_dir / filename}")
                    masks_written += 1

            except (BadZipFile, KeyError, OSError, ValueError) as error:
                print(f"Skipped {dicom_file}: {error}")

    print(f"Wrote {masks_written} masks to {mask_dir}")

# \/\/\/ frame isolation and saving \/\/\/ #
def save_annotated_frames(csv_path, mask_dir, frame_dir):
    set_csv_field_size_limit()
    frame_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        key = lambda row: (row["zip_path"], row["json_member"], row["dicom_file"])

        for (zip_path, _json_member, dicom_file), grouped_rows in itertools.groupby(
            reader, key=key
        ):
            rows = list(grouped_rows)

            relative_zip = Path(zip_path).relative_to(ROOT_DIR)
            subfolder = relative_zip.parent / relative_zip.stem
            source_stem = Path(dicom_file).stem

            annotated_frames = {
                int(row["frame_index"])
                for row in rows
                if (mask_dir / subfolder /
                    f"{source_stem}_frame_{int(row['frame_index']):04d}.png").is_file()
            }
            if not annotated_frames:
                continue

            with ZipFile(zip_path) as archive:
                with archive.open(dicom_file) as stream:
                    dicom = pydicom.dcmread(stream)

            pixels = dicom.pixel_array
            frame_count = int(getattr(dicom, "NumberOfFrames", 1))
            output_dir = frame_dir / subfolder
            output_dir.mkdir(parents=True, exist_ok=True)

            for frame_index in sorted(annotated_frames):
                if not 0 <= frame_index < frame_count:
                    print(f"Skipping invalid frame {frame_index}: {dicom_file}")
                    continue

                image = pixels[frame_index] if frame_count > 1 else pixels

                # pydicom = RGB; OpenCV = BGR for PNG output
                if image.ndim == 3 and image.shape[-1] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                output_path = output_dir / (
                    f"{source_stem}_frame_{frame_index:04d}.png"
                )
                if not cv2.imwrite(str(output_path), image):
                    print(f"Could not write frame: {output_path}")


output_csv = Path("D:/Uni work/Year 3/Thrombus+/Internal (Post June)/SegmentationModel/annotation_metadata.csv")
    # ==== 2. Uncomment to create annotation_metadata.csv used by create_gt_masks ====
#ANN_DIR = (get_json_path(ROOT_DIR))
#metadata = read_json(ANN_DIR)
#write_to_csv(output_csv, metadata)
#print(f"Wrote {len(metadata)} annotation rows to {output_csv}")

    # ==== 3. Uncomment to create and save masks and label_map.csv ====
#create_gt_masks(output_csv, output_csv.parent / "gt_masks")     # correct GR2's file structure to prevent error

    # ==== 4. Uncomment to save frames corresponding to masks from annotation_metadata.csv ====
save_annotated_frames(output_csv, output_csv.parent / "gt_masks", output_csv.parent / "dicom_frames",)