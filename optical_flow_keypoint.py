import os
import json
import cv2
import numpy as np



def extract_keypoints(json_data):
    """
    Extracts keypoint coordinates from JSON in format (N, 1, 2) float32 for LK optical flow.
    """
    keypoints_list = []

    if json_data.get("objects"):
        obj = json_data["objects"][0]
        for kp in obj.get("keypoints", []):
            if "projected_location" in kp:
                x, y = kp["projected_location"]
                keypoints_list.append([x, y])

    keypoints_np = np.array(keypoints_list, dtype=np.float32)
    return keypoints_np.reshape(-1, 1, 2)

def draw_sparse_flow(img, prev_points, next_points, color=(0, 255, 0)):
    """
    Draws optical flow lines between prev and next keypoints.
    prev_points, next_points: (N, 2)
    """
    mask = np.zeros_like(img)

    for (new, old) in zip(next_points, prev_points):
        a, b = new.astype(int)
        c, d = old.astype(int)

        mask = cv2.line(mask, (a, b), (c, d), color, 2)
        img = cv2.circle(img, (a, b), 5, color, -1)

    return cv2.add(img, mask)


# ------------------------------------------------------------
# 3. Main Optical Flow Pipeline
# ------------------------------------------------------------
def calculate_and_save_sparse_flow(input_dir, output_dir):

    # CONFIG
    BLUR_KERNEL = (5, 5)
    # More robust LK settings
    LK_WIN = (21, 21)
    LK_LEVELS = 2
    LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    FLOW_COLOR = (0, 69, 255)
    START_INDEX = 1

    # Output directories
    VISUAL_DIR = os.path.join(output_dir, "flow_visualizations")
    DATA_DIR = os.path.join(output_dir, "flow_data")
    os.makedirs(VISUAL_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Numerically sort .jpg files (safe against names like "000900.rgb.jpg")
    def numeric_key(fname):
        # take first numeric token from filename
        try:
            return int(fname.split('.')[0])
        except:
            # fallback to lexicographic if weird name
            return fname

    image_files = sorted(
        [f for f in os.listdir(input_dir) if f.lower().endswith(".jpg")],
        key=numeric_key
    )

    if len(image_files) < 2:
        print("Need at least two frames to compute optical flow.")
        return

    prev_idx = START_INDEX - 1
    prev_image_name = image_files[prev_idx]
    prev_frame_path = os.path.join(input_dir, prev_image_name)

    if not os.path.exists(prev_frame_path):
        print(f"Missing initial frame: {prev_frame_path}")
        return

    prev_frame = cv2.imread(prev_frame_path)
    if prev_frame is None:
        print(f"Failed to load image {prev_frame_path}")
        return

    # Load JSON for initial frame (use split()[0] to handle extra dots)
    base = prev_image_name.split('.')[0]
    json_name = base + ".json"
    json_path = os.path.join(input_dir, json_name)

    try:
        with open(json_path, "r") as f:
            json_data = json.load(f)
        points_to_track = extract_keypoints(json_data)
    except Exception as e:
        print(f" JSON load error for {json_path}: {e}")
        return

    if points_to_track.size == 0:
        print(f" No keypoints found in {json_path}. Cannot start tracking.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, BLUR_KERNEL, 0)

    print(f"➡ Initialized tracking with {len(points_to_track)} points at frame {prev_idx}")

    # helper to clamp/filter points inside image
    def filter_points_inside(pts, img_shape):
        h, w = img_shape
        pts2 = pts.reshape(-1, 2)
        mask = (pts2[:, 0] >= 0) & (pts2[:, 0] < w) & (pts2[:, 1] >= 0) & (pts2[:, 1] < h)
        filtered = pts2[mask]
        return filtered.reshape(-1, 1, 2).astype(np.float32)

    # MAIN LOOP
    for i in range(START_INDEX, len(image_files)):

        image_name = image_files[i]
        curr_frame_path = os.path.join(input_dir, image_name)
        curr_frame = cv2.imread(curr_frame_path)
        if curr_frame is None:
            print(f"⚠ Skipping unreadable image: {curr_frame_path}")
            continue

        next_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.GaussianBlur(next_gray, BLUR_KERNEL, 0)

        # ensure points are inside prev frame boundaries
        if points_to_track is None or len(points_to_track) == 0:
            print(f"⚠ No points left to track at frame {i}. Attempting re-init from JSON.")
            # attempt reinitialization from current frame's JSON
            base = image_name.split('.')[0]
            json_path = os.path.join(input_dir, base + ".json")
            try:
                with open(json_path, 'r') as jf:
                    jd = json.load(jf)
                    new_pts = extract_keypoints(jd)
                    # filter to be safe
                    new_pts = filter_points_inside(new_pts, next_gray.shape)
                    if new_pts.size > 0:
                        points_to_track = new_pts
                        prev_gray = next_gray
                        print(f"↺ Reinitialized {len(points_to_track)} points from {json_path} at frame {i}")
                        continue
                    else:
                        print(f" Reinit JSON had zero valid points in-image at frame {i}.")
                        break
            except Exception as e:
                print(f"Reinit JSON read error {json_path}: {e}")
                break

        # Always filter points to be inside image before calling LK
        points_to_track = filter_points_inside(points_to_track, prev_gray.shape)

        if points_to_track.size == 0:
            print(f"⚠ After filtering, no points in-bounds at frame {i}. Attempting re-init.")
            # try re-init as above
            base = image_name.split('.')[0]
            json_path = os.path.join(input_dir, base + ".json")
            try:
                with open(json_path, 'r') as jf:
                    jd = json.load(jf)
                    new_pts = extract_keypoints(jd)
                    new_pts = filter_points_inside(new_pts, next_gray.shape)
                    if new_pts.size > 0:
                        points_to_track = new_pts
                        prev_gray = next_gray
                        print(f"↺ Reinitialized {len(points_to_track)} points from {json_path} at frame {i}")
                        continue
                    else:
                        print(f"Reinit JSON had zero valid points in-image at frame {i}.")
                        break
            except Exception as e:
                print(f"Reinit JSON read error {json_path}: {e}")
                break

        # Compute LK optical flow
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            prev_gray,
            next_gray,
            points_to_track,
            None,
            winSize=LK_WIN,
            maxLevel=LK_LEVELS,
            criteria=LK_CRITERIA
        )

        # Try normal LK flow
        try:
            status = status.reshape(-1)
        except:
            status = None

        # If LK fails OR loses points → hard reset using current JSON
        need_reset = False
        if next_pts is None or status is None:
            need_reset = True
        else:
            good = status.sum()
            if good < len(points_to_track):   # lost even one keypoint → reset
                need_reset = True

        # ---- HARD RESET LOGIC ----
        if need_reset:
            base = image_name.split('.')[0]
            json_path = os.path.join(input_dir, base + ".json")

            try:
                with open(json_path, "r") as f:
                    jd = json.load(f)

                json_pts = extract_keypoints(jd)  # (7,1,2)
                json_pts = filter_points_inside(json_pts, next_gray.shape)

                if json_pts.size > 0:
                    points_to_track = json_pts
                    prev_gray = next_gray
                    print(f"HARD RESET at frame {i}")
                    continue
                else:
                    print(f"JSON for frame {i} has no valid points. Stopping.")
                    break

            except Exception as e:
                print(f"Failed to load JSON for reset at frame {i}: {e}")
                break

        # ---- Normal flow when NO RESET triggered ----
        # status == 1 → good points
        good_prev = points_to_track[status == 1].reshape(-1, 2)
        good_next = next_pts[status == 1].reshape(-1, 2)

        # Save flow
        flow_vectors = (good_next - good_prev).astype(np.float16)
        flow_coords = good_prev.astype(np.uint16)

        np.savez_compressed(
            os.path.join(DATA_DIR, f"flow_sparse_{i:04d}.npz"),
            COORDS=flow_coords,
            VECTORS=flow_vectors
        )

        # Visualize
        vis_img = draw_sparse_flow(
            curr_frame.copy(),
            good_prev,
            good_next,
            color=FLOW_COLOR
        )
        cv2.imwrite(os.path.join(VISUAL_DIR, f"flow_visual_{i:04d}.jpg"), vis_img)

        # UPDATE
        prev_gray = next_gray
        points_to_track = good_next.reshape(-1, 1, 2)

    print("Completed sparse optical flow tracking.")
    print(f"Flow data saved in: {DATA_DIR}")
    print(f"Visualizations saved in: {VISUAL_DIR}")


# Example usage
INPUT_FOLDER = "ppanda-orb"
OUTPUT_FOLDER = "data_flow/real/panda-orb"
calculate_and_save_sparse_flow(INPUT_FOLDER, OUTPUT_FOLDER)