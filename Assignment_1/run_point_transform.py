import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image

# Point-guided image deformation
def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """
    Return
    ------
        A deformed image.
    """

    if image is None:
        return None

    warped_image = np.array(image)
    num_pairs = min(len(source_pts), len(target_pts))
    if num_pairs == 0:
        return warped_image

    # Use only valid source-target pairs.
    source_pts = np.asarray(source_pts[:num_pairs], dtype=np.float32)
    target_pts = np.asarray(target_pts[:num_pairs], dtype=np.float32)

    h, w = warped_image.shape[:2]

    # Build inverse warp (target -> source) so remap can sample without holes.
    control_points = target_pts
    displacements = source_pts - target_pts

    # Adaptive Gaussian RBF width from control-point spacing.
    if num_pairs > 1:
        pair_diff = control_points[:, None, :] - control_points[None, :, :]
        pair_dist = np.sqrt(np.sum(pair_diff * pair_diff, axis=2))
        valid_dist = pair_dist[pair_dist > eps]
        base_sigma = np.median(valid_dist) if valid_dist.size > 0 else max(h, w) * 0.25
    else:
        base_sigma = max(h, w) * 0.25
    sigma = max(base_sigma * max(alpha, eps), 1.0)

    # Solve RBF weights: K @ W = displacement
    ctrl_diff = control_points[:, None, :] - control_points[None, :, :]
    ctrl_sqdist = np.sum(ctrl_diff * ctrl_diff, axis=2)
    K = np.exp(-ctrl_sqdist / (2.0 * sigma * sigma))
    K += eps * np.eye(num_pairs, dtype=np.float32)

    try:
        weights = np.linalg.solve(K, displacements)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(K, displacements, rcond=None)[0]

    # Evaluate deformation field in chunks to control memory usage.
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    query_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    mapped_points = np.empty_like(query_points, dtype=np.float32)

    batch_size = 50000
    for start in range(0, query_points.shape[0], batch_size):
        end = min(start + batch_size, query_points.shape[0])
        q = query_points[start:end]
        q_diff = q[:, None, :] - control_points[None, :, :]
        q_sqdist = np.sum(q_diff * q_diff, axis=2)
        phi = np.exp(-q_sqdist / (2.0 * sigma * sigma))
        q_disp = phi @ weights
        mapped_points[start:end] = q + q_disp

    map_x = mapped_points[:, 0].reshape(h, w).astype(np.float32)
    map_y = mapped_points[:, 1].reshape(h, w).astype(np.float32)

    warped_image = cv2.remap(
        warped_image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101
    )

    return warped_image

def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()
