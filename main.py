import json
import argparse
from kproducts_dataset import KProductsDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--conf", default="./conf.json", type=str, help="Configuration file path.")
    # parser.add_argument("--mode", default="none", type=str, help="(none, refresh, convert_img)")
    parser.add_argument("--refresh-annot", default=False, action='store_true', help="Refresh Annotation")

    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        config = json.load(f)

    dataset = KProductsDataset(config, refresh_annot=args.refresh_annot)
