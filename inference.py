# image→Sam3dBody→MHR

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from notebook.utils import setup_sam_3d_body
from tools.vis_utils import visualize_sample_together

def lxlywh_to_xyxy(bbox):
    """
    Convert bbox from (lx, ly, w, h) to (x1, y1, x2, y2)

    Args:
        bbox: list / tuple / np.ndarray with 4 elements

    Returns:
        np.ndarray of shape (4,) -> [x1, y1, x2, y2]
    """
    lx, ly, w, h = bbox
    x1 = lx
    y1 = ly
    x2 = lx + w
    y2 = ly + h
    return np.array([x1, y1, x2, y2], dtype=np.float32)

def to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif obj is None:
        return None
    elif isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_serializable(v) for v in obj]
    else:
        return obj

def get_scene_id(image_name: str) -> str:
    """
    image_name: e.g. 'S094_1280x720_F000091_T015153.png'
    return: 'S094'
    """
    return image_name.split("_")[0]    

def main():

    # Set up the estimator
    estimator = setup_sam_3d_body(hf_repo_id="facebook/sam-3d-body-dinov3",
                                detector_name="vitdet",
                                detector_path="./checkpoints/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692",
                                segmentor_name="sam3", 
                                segmentor_path="./checkpoints/sam3/sam3.pt", 
                                fov_path="./checkpoints/moge-2-vitl-normal/model.pt", 
                                checkpoint_path="./checkpoints/sam-3d-body-dinov3/model.ckpt", 
                                mhr_path="./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt")

    # load PVCP2 label file
    pvcp_path = "./data/PVCP/"
    pvcp_frame = os.path.join(pvcp_path, 'frame')
    pvcp_label = os.path.join(pvcp_path, 'annotation', 'dataset_pose.json')

    with open(pvcp_label, 'r') as f:
        pvcp_data = json.load(f)



    all_json_dir = os.path.join(pvcp_path, "sam3dbody_output", "all_json")
    os.makedirs(all_json_dir, exist_ok=True)

    scene_npy_dir = os.path.join(pvcp_path, "sam3dbody_output", "scene_npy")
    os.makedirs(scene_npy_dir, exist_ok=True)

    scene_json_dir = os.path.join(pvcp_path, "sam3dbody_output", "scene_json")
    os.makedirs(scene_json_dir, exist_ok=True)

    # 每个 scene 累积一个列表，最后再 stack 成 (N, D)
    scene_npy_buffer = defaultdict(lambda: {
        "lbs_model_params": [],
        "identity_coeffs": [],
        "face_expr_coeffs": [],
    })

    scene_json_buffer = defaultdict(dict)

    items = list(pvcp_data.items())
    items = sorted(items, key=lambda x: x[0])

    for frame, label in tqdm(items):
    # for frame, label in tqdm(pvcp_data.items()):
        bbox = pvcp_data[frame]['bbox']
        if bbox:
            image_name = pvcp_data[frame]['image_name']
            frame_path = os.path.join(pvcp_frame, frame)
            # img_bgr = cv2.imread(frame_path)
            outputs = estimator.process_one_image(img=frame_path, bboxes=lxlywh_to_xyxy(bbox))
            
            # all_json -------------------------------------------
            json_outputs = to_json_serializable(outputs)
            output_json_name = f"{os.path.splitext(frame)[0]}_mhr.json"

            with open(os.path.join(all_json_dir, output_json_name), "w", encoding="utf-8") as f:
                json.dump(json_outputs, f, indent=2)
            

            # scene npy -------------------------------------------
            scene_id = get_scene_id(image_name)  # e.g. "S094"

            if outputs is None or len(outputs) == 0:
                continue

            for o in outputs:
                scene_npy_buffer[scene_id]["lbs_model_params"].append(o["mhr_model_params"])
                scene_npy_buffer[scene_id]["identity_coeffs"].append(o["shape_params"])
                scene_npy_buffer[scene_id]["face_expr_coeffs"].append(o["expr_params"])     

            # scene json -------------------------------------------
            o = outputs[0]
            scene_json_buffer[scene_id][frame] = {
                "lbs_model_params": to_json_serializable(o["mhr_model_params"]),
                "identity_coeffs":  to_json_serializable(o["shape_params"]),
                "face_expr_coeffs": to_json_serializable(o["expr_params"]),
            }

    for scene_id, data in scene_npy_buffer.items():
        if len(data["lbs_model_params"]) == 0:
            continue

        lbs  = np.stack(data["lbs_model_params"], axis=0).astype(np.float32)   # (N, 204)
        iden = np.stack(data["identity_coeffs"], axis=0).astype(np.float32)    # (N, 45)
        expr = np.stack(data["face_expr_coeffs"], axis=0).astype(np.float32)   # (N, 72)

        save_scene_npy_path = os.path.join(scene_npy_dir, f"{scene_id}_mhr.npy")

        np.save(save_scene_npy_path, {
            "lbs_model_params": lbs,
            "identity_coeffs": iden,
            "face_expr_coeffs": expr,
        })

        print(f"✅ Saved {scene_id}: lbs{lbs.shape}, id{iden.shape}, expr{expr.shape} -> {save_scene_npy_path}")



    for scene_id, frame_map in scene_json_buffer.items():
        if len(frame_map) == 0:
            continue

        save_scene_json_path = os.path.join(scene_json_dir, f"{scene_id}_mhr.json")
        with open(save_scene_json_path, "w", encoding="utf-8") as f:
            json.dump(frame_map, f, indent=2)

        print(f"✅ Saved {scene_id} json with {len(frame_map)} frames -> {save_scene_json_path}")


if __name__ == '__main__':
    main()