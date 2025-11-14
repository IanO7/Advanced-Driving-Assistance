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
VIDEO_PATH_DEFAULT = "test_videos/pedestrian_crash.mp4"  # Default video when USE_CAMERA_DEFAULT is False

# Feature defaults when no --depth/--ldw flags are provided
ENABLE_DEPTH_DEFAULT = True
ENABLE_LDW_DEFAULT = True

# Output saving controls (for Run button / no-arg runs)
SAVE_OUTPUT_DEFAULT = False                 # True to save by default, False to not save by default
OUTPUT_PATH_DEFAULT = "output.mp4"         # Default filename when saving is enabled without --output

# Display toggles (set here for quick on/off without CLI flags)
SHOW_MAIN_WINDOW_DEFAULT = True             # Show combined LDW + depth window
SHOW_BIRDSEYE_DEFAULT = True                # Show Bird's Eye proximity window
ALERT_SOUND_ENABLED_DEFAULT = True          # Keep sound alerts (set False to mute)


def main():
    parser = argparse.ArgumentParser(description="Advanced Driving Assistance System")
    parser.add_argument('--depth', action='store_true', help='Enable depth estimation')
    parser.add_argument('--ldw', action='store_true', help='Enable lane departure warning')
    # Default is None so we can decide from simple toggles when no args are passed
    parser.add_argument('--video', type=str, default=None, help='Input source: path to a video file. To use a webcam, pass --camera or set --video to a numeric index like 0.')
    parser.add_argument('--camera', type=int, default=None, help='Camera index to use (e.g., 0 for default webcam). Overrides --video when provided.')
    parser.add_argument('--save', action='store_true', help='Save the visualized output to a video file (overrides default toggles).')
    parser.add_argument('--output', type=str, default=None, help='Output video file path. If not provided but --save is set, uses OUTPUT_PATH_DEFAULT from main.py.')
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

        # Decide output saving based on priority:
        # 1) If --output is provided -> save to that path
        # 2) Else if --save is provided -> save to OUTPUT_PATH_DEFAULT
        # 3) Else -> use SAVE_OUTPUT_DEFAULT toggle
        if args.output is not None:
            output_path = args.output
            print(f"[INFO] Saving output to '{output_path}' (from --output).")
        elif args.save:
            output_path = OUTPUT_PATH_DEFAULT
            print(f"[INFO] Saving output to '{output_path}' (from --save).")
        else:
            output_path = OUTPUT_PATH_DEFAULT if SAVE_OUTPUT_DEFAULT else None
            if output_path:
                print(f"[INFO] Saving output to '{output_path}' (from default toggle).")
            else:
                print("[INFO] Not saving output (from default toggle).")

        # You can add logic to pass args.ldw to MiDaS or handle overlays here
        midas.infer_video(
            source=source,
            output_path=output_path,
            display_main=SHOW_MAIN_WINDOW_DEFAULT,
            display_birdseye=SHOW_BIRDSEYE_DEFAULT,
            sound_enabled=ALERT_SOUND_ENABLED_DEFAULT
        )
    elif args.ldw:
        print("[INFO] Running only LDW overlay (example, not standalone)")
        # You would need to implement a video loop here if you want LDW only
        # For now, LDW is integrated in depth_estimation
        pass

if __name__ == "__main__":
    main()
