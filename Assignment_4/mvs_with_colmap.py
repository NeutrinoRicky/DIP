import os
import subprocess
import argparse
import shutil

# Allow COLMAP (Qt-based) to run on headless servers without an X display.
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def resolve_colmap_binary(explicit_path=None):
    if explicit_path:
        return explicit_path

    for candidate in ('colmap', 'COLMAP'):
        resolved = shutil.which(candidate)
        if resolved is not None:
            return resolved

    raise FileNotFoundError(
        "COLMAP executable was not found. Install COLMAP and add it to PATH, "
        "or pass --colmap_bin with the absolute path to COLMAP.bat / colmap.exe."
    )


def run_colmap(colmap_bin, *args):
    subprocess.run([colmap_bin, *args], check=True)


def get_colmap_help(colmap_bin, command):
    result = subprocess.run(
        [colmap_bin, command, '-h'],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout


def choose_gpu_flag(help_text, new_flag, legacy_flag):
    if new_flag in help_text:
        return new_flag
    if legacy_flag in help_text:
        return legacy_flag
    raise RuntimeError(
        f"Could not find either '{new_flag}' or '{legacy_flag}' in COLMAP help output."
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run COLMAP for multi-view stereo')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the input directory containing images in data_dir/images')
    parser.add_argument('--colmap_bin', type=str, default=None, help='Optional absolute path to COLMAP.bat or colmap.exe')
    args = parser.parse_args()
    data_dir = args.data_dir
    colmap_bin = resolve_colmap_binary(args.colmap_bin)
    feature_extractor_help = get_colmap_help(colmap_bin, 'feature_extractor')
    exhaustive_matcher_help = get_colmap_help(colmap_bin, 'exhaustive_matcher')
    feature_extraction_gpu_flag = choose_gpu_flag(
        feature_extractor_help,
        '--FeatureExtraction.use_gpu',
        '--SiftExtraction.use_gpu',
    )
    feature_matching_gpu_flag = choose_gpu_flag(
        exhaustive_matcher_help,
        '--FeatureMatching.use_gpu',
        '--SiftMatching.use_gpu',
    )

    # Feature extraction with shared intrinsics (assume it's the same camera)
    run_colmap(colmap_bin, 'feature_extractor', '--image_path', os.path.join(data_dir, 'images'), '--database_path', os.path.join(data_dir, 'database.db'), '--ImageReader.single_camera', '1', '--ImageReader.camera_model', 'PINHOLE', feature_extraction_gpu_flag, '0')

    # Feature matching
    run_colmap(colmap_bin, 'exhaustive_matcher', '--database_path', os.path.join(data_dir, 'database.db'), feature_matching_gpu_flag, '0')

    # Create sparse reconstruction folder
    os.makedirs(os.path.join(data_dir, 'sparse'), exist_ok=True)

    # Sparse reconstruction
    run_colmap(colmap_bin, 'mapper', '--image_path', os.path.join(data_dir, 'images'), '--database_path', os.path.join(data_dir, 'database.db'), '--output_path', os.path.join(data_dir, 'sparse'))

    # Convert binary model to text format
    os.makedirs(os.path.join(data_dir, 'sparse', '0_text'), exist_ok=True)
    run_colmap(colmap_bin, 'model_converter', '--input_path', os.path.join(data_dir, 'sparse', '0'), '--output_path', os.path.join(data_dir, 'sparse', '0_text'), '--output_type', 'TXT')

    print("COLMAP multi-view stereo pipeline completed successfully!")
    print("Sparse 3D reconstruction saved in:", os.path.join(data_dir, 'sparse', '0_text'))
    
