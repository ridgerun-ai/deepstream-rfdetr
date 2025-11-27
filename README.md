# DeepStream RF-DETR

> Run RF-DETR on NVIDIA DeepStream

This project provides the necessary parsing libraries and configuration files
that enable running RF-DETR models on NVIDIA DeepStream pipelines.

At the time being, the following features are supported:
- [x] RF-DETR Nano, Small, Medium, Large
- [x] FP32, FP16

The following features are a work in progress:
- [ ] INT8 calibration files
- [ ] RF-DETR for segmentation

## Supported DeepStream Versions

The project has been tested on the following DeepStream versions:
- DeepStream 8.0

## Building the Project

In a system with DeepStream installed:
```bash
make
```

This will generate:
```
libdeepstream-rfdetr.so
```

This is the library that must be configured in the `custom-lib-path` property of
the NvInfer.

## Downloading Pretrained Weights

The official RF-DETR weights are hosted by Roboflow and exposed via
ephemeral URLs. The recommended method is to use their `inference`
package to do so. We provide a small utility to download weights:

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you havent.

2. Download the weights by running:
```bash
uv run ./download_weights MODEL_ID
```
where `MODEL_ID` is one of:
- rfdetr-base (deprecated)
- rfdetr-nano
- rfdetr-small
- rfdetr-medium
- rfdetr-large

The script will download the weights to the current working directory
as `MODEL_ID.onnx`.

## Using RF-DETR in DeepStream

An example configuration file for NvInfer is provided in
[deepstream_rfdetr_bbox_config.txt](/deepstream_rfdetr_bbox_config.txt).

The specific fields that make RF-DETR work are:
- net-scale-factor
- offsets
- custom-lib-path
- parse-bbox-func-name
- onnx-file / model-engine-file
- num-detected-classes
- model-color-format
- network-type
- maintain-aspect-ratio
- cluster-mode
- network-input-order

This config file works fine with DeepStream sample apps. A very simple pipeline
that performs inference using RF-DETR over a file, and saves the result to a
file is:

```bash
gst-launch-1.0 -e filesrc location=INPUT.mp4 ! decodebin ! queue ! mux.sink_0 \
    nvstreammux name=mux width=1920 height=1080 batch-size=1 ! \
    nvinfer config-file-path=deepstream_rfdetr_bbox_config.txt ! \
    queue ! nvdsosd ! nvv4l2h264enc ! h264parse ! queue ! mp4mux ! \
    filesink location=OUTPUT.mp4
```

Remember to adjust the `nvstreammux` width and height properties to match the
image size of your input video.

## Performance 

TODO

## Notes for Maintainers

- Build the project as `make DEV=1`. This will build in debug mode, as
  well as not allowing warnings.
- Format the code by running `make format`.
- Lint the code by running `make lint`.
- Make sure all these three are clean before submitting a PR.
