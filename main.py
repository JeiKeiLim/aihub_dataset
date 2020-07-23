import json
import argparse
from dataset.kproducts_dataset import KProductsDataset
import numpy as np
import random

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--conf", default="./conf.json", type=str, help="Configuration file path.")
    parser.add_argument("--resize", default=False, action='store_true', help="Resize Image and Save to --target-path")
    parser.add_argument("--resize-no-copy-annotation", default=True, dest='resize_copy_annotation', action='store_false', help="Copy annotaion json file on resizing")
    parser.add_argument("--resize-num-cpus", default="1.0", type=str, help="Number(int) or proportion(float) of cpus to utilize in multiprocess. (Ex: 1 = One Core, 1.0 = All Core)")
    parser.add_argument("--resize-no-multi-process", dest="resize_multi_process", default=True, action='store_false', help="Use Multi Process on Resizing Images")
    parser.add_argument("--target-root", default="./export", type=str, help="Target Directory Path.")
    parser.add_argument("--skip-exists", default=False, action='store_true', help="Skip Process if file already exists")
    parser.add_argument("--target-w", default=320, type=int, help="Target width size for resizing images")
    parser.add_argument("--refresh-annot", default=False, action='store_true', help="Refresh Annotation.")
    parser.add_argument("--refresh-multi-process", default=False, action='store_true', help="Use Multi Process on Refreshing Annotations")
    parser.add_argument("--vectorize", default=False, action='store_true', help="Vecotorize Images by output of ImageNet Pre-trained Model for further clustering purpose.")
    parser.add_argument("--vectorize-model", default="resnet152v2", help="Possible values: (densenet121, densenet169, densenet201, inceptionresnetv2, inceptionv3, mobilenet, mobilenetv2, nasnetlarge, nasnetmobile, resnet101, resnet101v2, resnet152, resnet152v2, resnet50, resnet50v2, vgg16, vgg19, xception)")
    parser.add_argument("--model-input-w", default=224, type=int, help="Vectorize Model Input Width")
    parser.add_argument("--model-input-h", default=224, type=int, help="Vectorize Model Input Height")
    parser.add_argument("--vectorize-multi-process", default=False, action='store_true', help="Use Multi Process on Vectorization.")
    parser.add_argument("--batch-size", default=32, type=int, help="Batch Size")
    parser.add_argument("--cluster", default=False, action='store_true', help="Perform Clustering in vectorization features")
    parser.add_argument("--eps", default=0.1, type=float, help="Epsilon Parameter for DBSCAN Clustering")
    parser.add_argument("--show-cluster-plot", default=False, action='store_true', help="Plot Cluster Result")
    parser.add_argument("--save-plot", default=False, action='store_true', help="Save Clustered Result Image")
    parser.add_argument("--reconstruct", default=False, action='store_true', help="Reconstruct dataset from clustering result")
    parser.add_argument("--reconstruct-root", default="./export", type=str, help="Reconstruct dataset root directory")
    parser.add_argument("--reconstruct-annotation-name", default="converted_annotation.csv", type=str, help="Reconstruct dataset annotation file name")
    parser.add_argument("--no-include-non-core", dest="include_non_core", default=True, action='store_false', help="Whether including non-core clustering index")
    parser.add_argument("--split-train-test", default=False, action='store_true', help="Splitting training and test set (Annotation only)")
    parser.add_argument("--split-balance-classes", default=False, action='store_true', help="Balancing Class Labels while splitting training and test set")
    parser.add_argument("--split-train-ratio", default=0.7, type=float, help="Splitting Training dataset ratio")
    parser.add_argument("--seed", default=7777, type=int, help="Seed value to match random numbers")

    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    with open(args.conf, 'r', encoding='UTF8') as f:
        config = json.load(f)
        config['self_path'] = args.conf

    dataset = KProductsDataset(config, refresh_annot=args.refresh_annot, refresh_multi_process=args.refresh_multi_process, seed=args.seed)
    if args.resize:
        if args.resize_num_cpus.find(".") >= 0:
            num_cpus = float(args.resize_num_cpus)
        else:
            num_cpus = int(args.resize_num_cpus)

        dataset.resize_dataset(target_w=args.target_w, target_root=args.target_root, skip_exists=args.skip_exists,
                               multiprocess=args.resize_multi_process, num_cpus=num_cpus, copy_annotation=args.resize_copy_annotation)

    if args.vectorize or args.cluster or args.reconstruct:
        from dataset.cluster_dataset import ClusterData

        if args.vectorize_model not in ClusterData.name_to_model_dict.keys():
            print("Wrong Model Name!")
            parser.print_help()
            exit(0)

        base_model = ClusterData.name_to_model_dict[args.vectorize_model]
        cluster_data = ClusterData(dataset, model_input_size=(args.model_input_h, args.model_input_w), BaseModel=base_model, seed=args.seed)

        if args.vectorize:
            cluster_data.vectorize_dataset(multiprocess=args.vectorize_multi_process, batch_size=args.batch_size)

        if args.cluster:
            cluster_data.cluster_dataset(eps=args.eps, plot=args.show_cluster_plot, save_plot=args.save_plot)

        if args.reconstruct:
            cluster_data.reconstruct_from_cluster_result(target_root=args.reconstruct_root,
                                                         target_annotation_name=args.reconstruct_annotation_name,
                                                         include_non_core=args.include_non_core)
    if args.split_train_test:
        dataset.split_train_test(train_ratio=args.split_train_ratio, balance_class=args.split_balance_classes)
