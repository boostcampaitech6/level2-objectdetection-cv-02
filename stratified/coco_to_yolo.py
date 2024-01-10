import os
import json
import shutil
from tqdm import tqdm

def convert_coco_to_yolo_format(root_dir: str, json_file: str, save_dir: str):
    # Check directory
    try:
        assert os.path.exists(os.path.join(root_dir, save_dir, "images")) == True
        assert os.path.exists(os.path.join(root_dir, save_dir, "labels")) == True
    except:
        os.makedirs(os.path.join(root_dir, save_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(root_dir, save_dir, "labels"), exist_ok=True)
    finally:
        print("Finish make directory")
    # Load json
    with open(os.path.join(root_dir, json_file), 'r') as f:
        coco_json = json.load(f)
    anoots = coco_json["annotations"]
    print("Start converting...")
    for image in tqdm(sorted(coco_json["images"], key=lambda x: x["id"])):
    
        w, h, file_name, image_id = image["width"], image["height"], image["file_name"], image["id"]
        file_name = file_name.split("/")[1]
        # filtering annotations
        obj_candits = list(filter(lambda x: x["image_id"] == image_id, anoots))
        # Save txt format to train yolo
        with open(os.path.join(root_dir, save_dir, "labels", f"{file_name[:-4]}.txt"), "w") as f:
            for obj_candit in obj_candits:
                # x1 y1 w h -> cx cy w h               
                cat_id = obj_candit["category_id"]
                x1, y1, width, height = obj_candit["bbox"]
                scaled_cx, scaled_cy = (x1+width/2) / w, (y1+height/2) / h
                scaled_width, scaled_height = width / w, height / h
            f.write("%s %.3f %.3f %.3f %.3f\n" %(cat_id, scaled_cx, scaled_cy, scaled_width, scaled_height))
            f.close() # Copy image to new directory
        shutil.copy(os.path.join(root_dir, "train", file_name), os.path.join(root_dir, save_dir, "images", file_name))
    print("Finish converting...")

convert_coco_to_yolo_format("../../dataset", "train.json", "yolo_train")