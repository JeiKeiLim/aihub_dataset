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
    parser.add_argument("--vectorize", default=False, action='store_true', help="Vecotorize Images by output of ImageNet Pre-trained Model for further clustering purpose.")
    parser.add_argument("--vectorize-model", default="resnet152v2", help="Possible values: (densenet121, densenet169, densenet201, inceptionresnetv2, inceptionv3, mobilenet, mobilenetv2, nasnetlarge, nasnetmobile, resnet101, resnet101v2, resnet152, resnet152v2, resnet50, resnet50v2, vgg16, vgg19, xception)")
    parser.add_argument("--model-input-w", default=224, type=int, help="Vectorize Model Input Width")
    parser.add_argument("--model-input-h", default=224, type=int, help="Vectorize Model Input Height")
    parser.add_argument("--vectorize-multi-process", default=False, action='store_true', help="Use Multi Process on Vectorization.")
    parser.add_argument("--batch-size", default=32, type=int, help="Batch Size")


    args = parser.parse_args()

    with open(args.conf, 'r') as f:
        config = json.load(f)

    dataset = KProductsDataset(config, refresh_annot=args.refresh_annot, refresh_multi_process=args.refresh_multi_process)
    if args.resize:
        dataset.resize_dataset(target_w=args.target_w, target_root=args.target_root, skip_exists=args.skip_exists, multiprocess=args.resize_multi_process)

    if args.vectorize:
        from cluster_dataset import ClusterData
        if args.vectorize_model not in ClusterData.name_to_model_dict.keys():
            print("Wrong Model Name!")
            parser.print_help()
            exit(0)

        base_model = ClusterData.name_to_model_dict[args.vectorize_model]
        cluster_data = ClusterData(dataset, model_input_size=(args.model_input_h, args.model_input_w), BaseModel=base_model)
        cluster_data.vectorize_dataset(multiprocess=args.vectorize_multi_process, batch_size=args.batch_size)
