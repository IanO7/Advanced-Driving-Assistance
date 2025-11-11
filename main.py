import argparse

# Import your feature modules
from depth_estimation import MiDaS
from ldw import ldw_overlay


def main():
    parser = argparse.ArgumentParser(description="Advanced Driving Assistance System")
    parser.add_argument('--depth', action='store_true', help='Enable depth estimation')
    parser.add_argument('--ldw', action='store_true', help='Enable lane departure warning')
    parser.add_argument('--video', type=str, default='car_crash.mp4', help='Input video file or camera index')
    parser.add_argument('--output', type=str, default='output.mp4', help='Output video file')
    args = parser.parse_args()

    # If no features are specified, enable all by default
    if not args.depth and not args.ldw:
        args.depth = True
        args.ldw = True

    if args.depth:
        print("[INFO] Running depth estimation (and LDW if integrated)...")
        midas = MiDaS(model_type="DPT_Hybrid")
        # You can add logic to pass args.ldw to MiDaS or handle overlays here
        midas.infer_video(source=args.video, output_path=args.output, display=True)
    elif args.ldw:
        print("[INFO] Running only LDW overlay (example, not standalone)")
        # You would need to implement a video loop here if you want LDW only
        # For now, LDW is integrated in depth_estimation
        pass

if __name__ == "__main__":
    main()
