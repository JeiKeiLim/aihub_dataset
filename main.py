import json
import argparse
from kproducts_dataset import KProductsDataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--conf", default="./conf.json", type=str, help="Configuration file path.")
    parser.add_argument("--resize", default=False, action='store_true', help="Resize Image and Save to --target-path")
    parser.add_argument("--target-root", default="./export", type=str, help="Target Directory Path.")
    parser.add_argument("--skip-exists", default=False, action='store_true', help="Skip Process if file already exists")
    parser.add_argument("--target-w", default=320, type=int, help="Target width size for resizing images")
    parser.add_argument("--refresh-annot", default=False, action='store_true', help="Refresh Annotation.")
    parser.add_argument("--refresh-multi-process", default=False, action='store_true', help="Use Multi Process on Refreshing Annotations")
    parser.add_argument("--resize-no-multi-process", dest="resize_multi_process", default=True, action='store_false', help="Use Multi Process on Resizing Images")

    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        config = json.load(f)

    dataset = KProductsDataset(config, refresh_annot=args.refresh_annot, refresh_multi_process=args.refresh_multi_process)
    if args.resize:
        dataset.resize_dataset(target_w=args.target_w, target_root=args.target_root, skip_exists=args.skip_exists, multiprocess=args.resize_multi_process)


