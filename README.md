# aihub_dataset
## Reading KProduct dataset in http://aihub.or.kr

# Requirements
```
pandas>=1.0.5
matplotlib>=3.3.0
tqdm>=4.48.0
p_tqdm>=1.3.3

tensorflow>=2.2.0 (Optional. Required for clustering dataset only)
```

# conf.json
``` json
{
  "dataset_root": "/Users/jeikei/Documents/datasets/aihub_kproduct_resized",
  "annotation_path": "./converted_annotation.csv",
  "image_w": 540,
  "image_h":405,
  "class_key": "소분류"
}
```

# Example
## 1. Refresh/Generate converted annotation.csv
``` shell
python main.py --refresh-annot
```

## 2. Resize Entire Dataset
``` shell
python main.py --resize --target-root /resized/dataset/root
```
## 3. Clustering Dataset
### 3.1. Vecorize Images (Further Clustering Purpose)
``` shell
python main.py --vectorize --vectorize-model resnet152v2 --batch-size 32
```

### 3.2. Cluster Images
``` shell
python main.py --cluster --eps 0.1
```
- eps value might require to change

### 3.4. Reconstruct Dataset by Clustering Results
``` shell
python main.py --reconstruct --reconstruct-root /reconstruct/dataset/root
```

### 3.5. All at once
``` shell
python main.py --vectorize --vectorize-model resnet152v2 --batch-size 32 --cluster --eps 0.1 --reconstruct --reconstruct-root /reconstruct/dataset/root
```

# USAGE
```
usage: main.py [-h] [--conf CONF] [--resize] [--resize-no-copy-annotation]
               [--resize-num-cpus RESIZE_NUM_CPUS] [--resize-no-multi-process]
               [--target-root TARGET_ROOT] [--skip-exists]
               [--target-w TARGET_W] [--refresh-annot]
               [--refresh-multi-process] [--vectorize]
               [--vectorize-model VECTORIZE_MODEL]
               [--model-input-w MODEL_INPUT_W] [--model-input-h MODEL_INPUT_H]
               [--vectorize-multi-process] [--batch-size BATCH_SIZE]
               [--cluster] [--eps EPS] [--show-cluster-plot] [--save-plot]
               [--reconstruct] [--reconstruct-root RECONSTRUCT_ROOT]
               [--reconstruct-annotation-name RECONSTRUCT_ANNOTATION_NAME]
               [--no-include-non-core]
optional arguments:
  -h, --help            show this help message and exit
  --conf CONF           Configuration file path. (default: ./conf.json)
  --resize              Resize Image and Save to --target-path (default:
                        False)
  --resize-no-copy-annotation
                        Copy annotaion json file on resizing (default: True)
  --resize-num-cpus RESIZE_NUM_CPUS
                        Number(int) or proportion(float) of cpus to utilize in
                        multiprocess. (Ex: 1 = One Core, 1.0 = All Core)
                        (default: 1.0)
  --resize-no-multi-process
                        Use Multi Process on Resizing Images (default: True)
  --target-root TARGET_ROOT
                        Target Directory Path. (default: ./export)
  --skip-exists         Skip Process if file already exists (default: False)
  --target-w TARGET_W   Target width size for resizing images (default: 320)
  --refresh-annot       Refresh Annotation. (default: False)
  --refresh-multi-process
                        Use Multi Process on Refreshing Annotations (default:
                        False)
  --vectorize           Vecotorize Images by output of ImageNet Pre-trained
                        Model for further clustering purpose. (default: False)
  --vectorize-model VECTORIZE_MODEL
                        Possible values: (densenet121, densenet169,
                        densenet201, inceptionresnetv2, inceptionv3,
                        mobilenet, mobilenetv2, nasnetlarge, nasnetmobile,
                        resnet101, resnet101v2, resnet152, resnet152v2,
                        resnet50, resnet50v2, vgg16, vgg19, xception)
                        (default: resnet152v2)
  --model-input-w MODEL_INPUT_W
                        Vectorize Model Input Width (default: 224)
  --model-input-h MODEL_INPUT_H
                        Vectorize Model Input Height (default: 224)
  --vectorize-multi-process
                        Use Multi Process on Vectorization. (default: False)
  --batch-size BATCH_SIZE
                        Batch Size (default: 32)
  --cluster             Perform Clustering in vectorization features (default:
                        False)
  --eps EPS             Epsilon Parameter for DBSCAN Clustering (default: 0.1)
  --show-cluster-plot   Plot Cluster Result (default: False)
  --save-plot           Save Clustered Result Image (default: False)
  --reconstruct         Reconstruct dataset from clustering result (default:
                        False)
  --reconstruct-root RECONSTRUCT_ROOT
                        Reconstruct dataset root directory (default: ./export)
  --reconstruct-annotation-name RECONSTRUCT_ANNOTATION_NAME
                        Reconstruct dataset annotation file name (default:
                        converted_annotation.csv)
  --no-include-non-core
                        Whether including non-core clustering index (default:
                        True)

```


