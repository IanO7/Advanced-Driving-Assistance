import argparse

# Import your feature modules
from depth_estimation import MiDaS
from ldw import ldw_overlay

# ================================
# Simple toggles for Run button use
# Change these values and just press Run (F5) with no arguments
# CLI flags (e.g., --camera, --video) will always override these defaults.
# ================================
USE_CAMERA_DEFAULT = False           # True: use webcam by default, False: use video file by default
CAMERA_INDEX_DEFAULT = 0             # Which camera index to use when USE_CAMERA_DEFAULT is True
VIDEO_PATH_DEFAULT = "test_videos/california_drive.mp4"  # Default video when USE_CAMERA_DEFAULT is False

# Feature defaults when no --depth/--ldw flags are provided
ENABLE_DEPTH_DEFAULT = True
ENABLE_LDW_DEFAULT = True


def main():
    parser = argparse.ArgumentParser(description="Advanced Driving Assistance System")
    parser.add_argument('--depth', action='store_true', help='Enable depth estimation')
    parser.add_argument('--ldw', action='store_true', help='Enable lane departure warning')
    # Default is None so we can decide from simple toggles when no args are passed
    parser.add_argument('--video', type=str, default=None, help='Input video file path. If you want to use a camera, pass --camera or set --video to a numeric index like 0.')
    parser.add_argument('--camera', type=int, default=None, help='Camera index to use (e.g., 0 for default webcam). Overrides --video when provided.')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video file')
    args = parser.parse_args()

    # If no features are specified, use simple defaults
    if not args.depth and not args.ldw:
        args.depth = ENABLE_DEPTH_DEFAULT
        args.ldw = ENABLE_LDW_DEFAULT

    if args.depth:
        print("[INFO] Running depth estimation (and LDW if integrated)...")
        midas = MiDaS(model_type="DPT_Hybrid")

        # Determine source in this priority:
        # 1) --camera if provided
        # 2) --video if provided (numeric -> camera index)
        # 3) Simple toggles above (USE_CAMERA_DEFAULT, etc.)
        source = None
        if args.camera is not None:
            source = int(args.camera)
            print(f"[INFO] Using camera index {source} (from --camera).")
        elif args.video is not None:
            source = args.video
            if isinstance(source, str) and source.isdigit():
                source = int(source)
                print(f"[INFO] Detected numeric --video value. Using camera index {source}.")
            else:
                print(f"[INFO] Using video file '{source}' (from --video).")
        else:
            # Fall back to simple toggles for no-arg Run button usage
            if USE_CAMERA_DEFAULT:
                source = int(CAMERA_INDEX_DEFAULT)
                print(f"[INFO] No input args provided. Using default camera index {source} (from toggles).")
            else:
                source = VIDEO_PATH_DEFAULT
                print(f"[INFO] No input args provided. Using default video '{source}' (from toggles).")

        # You can add logic to pass args.ldw to MiDaS or handle overlays here
        midas.infer_video(source=source, output_path=args.output, display=True)
    elif args.ldw:
        print("[INFO] Running only LDW overlay (example, not standalone)")
        # You would need to implement a video loop here if you want LDW only
        # For now, LDW is integrated in depth_estimation
        pass

if __name__ == "__main__":
    main()
