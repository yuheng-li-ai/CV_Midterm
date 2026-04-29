from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="CV HW2 unified harness")
    sub = parser.add_subparsers(dest="command", required=True)

    p_cls = sub.add_parser("train-classification", help="train Oxford-IIIT Pet classifier")
    p_cls.add_argument("--config", required=True)

    p_cls_analyze = sub.add_parser("analyze-classification", help="make confusion matrix and mistake grid for a classifier")
    p_cls_analyze.add_argument("--config", required=True)
    p_cls_analyze.add_argument("--checkpoint", required=True)
    p_cls_analyze.add_argument("--split", choices=["val", "test"], default="test")
    p_cls_analyze.add_argument("--out-dir", required=True)
    p_cls_analyze.add_argument("--max-examples", type=int, default=24)

    p_seg = sub.add_parser("train-segmentation", help="train Oxford-IIIT Pet U-Net segmenter")
    p_seg.add_argument("--config", required=True)

    p_convert = sub.add_parser("convert-visdrone", help="convert VisDrone DET annotations to YOLO format")
    p_convert.add_argument("--source", required=True)
    p_convert.add_argument("--out", required=True)

    p_det = sub.add_parser("train-detection", help="train YOLO detector on VisDrone")
    p_det.add_argument("--config", required=True)

    p_track = sub.add_parser("track-video", help="run detection tracking and line counting on a video")
    p_track.add_argument("--weights", required=True)
    p_track.add_argument("--video", required=True)
    p_track.add_argument("--out", required=True)
    p_track.add_argument("--line", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"), required=True)
    p_track.add_argument("--tracker", default="bytetrack.yaml")
    p_track.add_argument("--conf", type=float, default=0.25)
    p_track.add_argument("--iou", type=float, default=0.5)
    p_track.add_argument("--export-frames", nargs="*", type=int, default=[])

    args = parser.parse_args()

    if args.command == "train-classification":
        from cvhw2.classification.train import train_classification

        train_classification(args.config)
    elif args.command == "analyze-classification":
        from cvhw2.classification.analyze import analyze_classification

        analyze_classification(
            config_path=args.config,
            checkpoint=args.checkpoint,
            split=args.split,
            out_dir=args.out_dir,
            max_examples=args.max_examples,
        )
    elif args.command == "train-segmentation":
        from cvhw2.segmentation.train import train_segmentation

        train_segmentation(args.config)
    elif args.command == "convert-visdrone":
        from cvhw2.detection.visdrone import convert_visdrone

        yaml_path = convert_visdrone(args.source, args.out)
        print(f"Wrote {yaml_path}")
    elif args.command == "train-detection":
        from cvhw2.detection.train import train_detection

        train_detection(args.config)
    elif args.command == "track-video":
        from cvhw2.detection.track import track_video

        track_video(
            weights=args.weights,
            video=args.video,
            out=args.out,
            line=tuple(args.line),
            tracker=args.tracker,
            conf=args.conf,
            iou=args.iou,
            export_frames=args.export_frames,
        )


if __name__ == "__main__":
    main()
