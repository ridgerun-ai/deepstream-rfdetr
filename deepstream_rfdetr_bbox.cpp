/* Copyright (C) 2025 RidgeRun, LLC <support@ridgerun.ai>
 * All Rights Reserved.
 *
 * The contents of this software are proprietary and confidential to
 * RidgeRun, LLC. No part of this program may be photocopied,
 * reproduced or translated into another programming language without
 * prior written consent of RidgeRun, LLC. The user is free to modify
 * the source code after obtaining a software license from
 * RidgeRun. All source code changes must be provided back to RidgeRun
 * without any encumbrance.
 */

#include <nvdsinfer.h>
#include <nvdsinfer_custom_impl.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct Layer {
  struct Classes {
    static constexpr std::string_view NAME = "labels";
    static constexpr NvDsInferDataType TYPE = FLOAT;

    enum Dims : std::uint8_t {
      DETECTIONS,
      CLASSES,
      NUM_DIMS,
    };

    static constexpr unsigned int BACKGROUND = 0;
  };

  struct Boxes {
    static constexpr std::string_view NAME = "dets";
    static constexpr NvDsInferDataType TYPE = FLOAT;

    enum Dims : std::uint8_t {
      DETECTIONS,
      BOXES,
      NUM_DIMS,
    };

    enum Box : std::uint8_t { CX, CY, W, H, SIZE };
  };
};

template <typename T>
auto softmax_of_best_logit(std::span<const T> logit,
                           std::span<const T> thresholds)
    -> std::optional<std::pair<std::size_t, T>> {
  const std::size_t size = logit.size();
  assert(size > 0);
  assert(thresholds.size() == size);

  // 0) argmax
  std::size_t max_idx = 0;
  T max_val = logit[0];
  for (std::size_t i = 1; i < size; ++i) {
    if (logit[i] > max_val) {
      max_val = logit[i];
      max_idx = i;
    }
  }

  // 1) Ignore if its background
  if (max_idx == Layer::Classes::BACKGROUND) {
    return std::nullopt;
  }

  // 2) extract the threshold for this class
  T threshold = thresholds[max_idx];

  // 3) threshold → limit
  const long double eps = std::numeric_limits<long double>::epsilon();
  const long double thr = std::max(static_cast<long double>(threshold), eps);

  const long double limit = 1.0L / thr - 1.0L;

  // 4) accumulate Σ_{j≠max} exp(z_j - z_max)
  long double sum_exp_others = 0.0L;

  for (std::size_t i = 0; i < size; ++i) {
    if (i == max_idx) {
      continue;
    }

    const auto diff = static_cast<long double>(logit[i] - max_val);
    sum_exp_others += std::exp(diff);

    if (sum_exp_others > limit) {
      return std::nullopt;
    }
  }

  // 5) exact softmax probability of max logit
  const long double p_max_ld = 1.0L / (1.0L + sum_exp_others);
  const T p_max = static_cast<T>(p_max_ld);

  return std::make_pair(max_idx, p_max);
}

auto find_layer(const std::vector<NvDsInferLayerInfo> &layers,
                const std::string_view &name,
                NvDsInferDataType type) -> std::optional<NvDsInferLayerInfo> {
  auto name_and_type_match = [&](auto const &layer) -> bool {
    return layer.dataType == type && layer.layerName == name;
  };

  auto ilayer = std::find_if(layers.begin(), layers.end(), name_and_type_match);

  if (ilayer == layers.end()) {
    return std::nullopt;
  }

  return *ilayer;
};

template <typename T>
auto view(std::span<const T> buffer, unsigned int offset,
          unsigned int size) -> std::span<const T> {
  const auto block_start = static_cast<std::size_t>(offset) * size;
  const auto block_size = static_cast<std::size_t>(size);

  assert(block_start + block_size <= buffer.size());

  return buffer.subspan(block_start, block_size);
}

template <typename T>
auto parse_detection(std::span<const T> boxes, std::span<const T> classes,
                     const NvDsInferParseDetectionParams &params,
                     unsigned int width, unsigned int height)
    -> std::optional<NvDsInferObjectDetectionInfo> {
  auto best = softmax_of_best_logit(
      classes, std::span{params.perClassPreclusterThreshold});
  if (!best) {
    return std::nullopt;
  }

  T box_x1 =
      (boxes[Layer::Boxes::Box::CX] - boxes[Layer::Boxes::Box::W] / 2) * width;
  T box_y1 =
      (boxes[Layer::Boxes::Box::CY] - boxes[Layer::Boxes::Box::H] / 2) * height;
  T box_x2 = box_x1 + boxes[Layer::Boxes::Box::W] * width;
  T box_y2 = box_y1 + boxes[Layer::Boxes::Box::H] * height;

  const float max_x = static_cast<float>(width) - 1.0F;
  const float max_y = static_cast<float>(height) - 1.0F;
  constexpr float min_x = 0.0F;
  constexpr float min_y = 0.0F;

  box_x1 = std::clamp(box_x1, min_x, max_x);
  box_y1 = std::clamp(box_y1, min_y, max_y);
  box_x2 = std::clamp(box_x2, min_x, max_x);
  box_y2 = std::clamp(box_y2, min_y, max_y);

  NvDsInferObjectDetectionInfo pred;
  pred.classId = best->first;
  pred.detectionConfidence = best->second;
  pred.left = box_x1;
  pred.top = box_y1;
  pred.width = box_x2 - box_x1;
  pred.height = box_y2 - box_y1;

  return pred;
}

template <typename T>
auto layer_to_span(const NvDsInferLayerInfo &layer) -> std::span<const T> {
  std::size_t layer_size = 1;
  for (unsigned int i = 0; i < layer.inferDims.numDims; i++) {
    layer_size *= layer.inferDims.d[i];
  }

  return std::span<const T>(static_cast<T *>(layer.buffer), layer_size);
}

}  // namespace

extern "C" auto deepstream_rfdetr_bbox(
    const std::vector<NvDsInferLayerInfo> &layers,
    const NvDsInferNetworkInfo &network,
    const NvDsInferParseDetectionParams &params,
    std::vector<NvDsInferObjectDetectionInfo> &detections) -> bool {
  auto layer_boxes = find_layer(layers, Layer::Boxes::NAME, Layer::Boxes::TYPE);
  auto layer_classes =
      find_layer(layers, Layer::Classes::NAME, Layer::Classes::TYPE);

  if (!layer_boxes || !layer_classes) {
    std::cerr << "DeepStream-RFDETR: Unable to find output layers named \""
              << Layer::Boxes::NAME << "\" and \"" << Layer::Classes::NAME
              << "\". Did you pass the right engine?\n"
              << "The output layer names are: \n";
    std::for_each(layers.begin(), layers.end(), [](const auto &layer) {
      std::cerr << "\t- " << layer.layerName << "\n";
    });

    return false;
  }

  auto layer_classes_num_dims = layer_classes->inferDims.numDims;
  auto layer_boxes_num_dims = layer_classes->inferDims.numDims;

  if (Layer::Classes::Dims::NUM_DIMS != layer_classes_num_dims ||
      Layer::Boxes::Dims::NUM_DIMS != layer_boxes_num_dims) {
    std::cerr << "DeepStream-RFDETR: layer number of dimensions don't match. "
                 "Did you pass in the correct model?\n"
              << "\t- " << Layer::Classes::NAME << ": "
              << Layer::Classes::Dims::NUM_DIMS << " (expected) <-> "
              << layer_classes_num_dims << " (got)\n"
              << "\t- " << Layer::Boxes::NAME << ": "
              << Layer::Boxes::Dims::NUM_DIMS << " (expected) <-> "
              << layer_boxes_num_dims << " (got)\n";
    return false;
  }

  const std::span<const unsigned int, NVDSINFER_MAX_DIMS> layer_boxes_dims{
      layer_boxes->inferDims.d};
  auto num_detections_boxes = layer_boxes_dims[Layer::Boxes::Dims::DETECTIONS];
  auto num_box_params = layer_boxes_dims[Layer::Boxes::Dims::BOXES];

  if (Layer::Boxes::Box::SIZE != num_box_params) {
    std::cerr << "DeepStream-RFDETR: The boxes tensor has a "
                 "different box dimension size ("
              << num_box_params << ") than the expected ("
              << Layer::Boxes::Box::SIZE << "). Did you pass "
              << "in the correct model?\n";
    return false;
  }

  const std::span<const unsigned int, NVDSINFER_MAX_DIMS> layer_classes_dims{
      layer_classes->inferDims.d};
  auto num_detections_classes =
      layer_classes_dims[Layer::Classes::Dims::DETECTIONS];
  auto num_classes = layer_classes_dims[Layer::Classes::Dims::CLASSES];

  if (params.numClassesConfigured != num_classes) {
    std::cerr << "DeepStream-RFDETR: The classes tensor has a "
                 "different dimension size ("
              << num_classes << ") than the expected ("
              << params.numClassesConfigured << "). Check your "
              << "nvinfer config file!\n";
    return false;
  }

  if (num_detections_boxes != num_detections_classes) {
    std::cerr << "DeepStream-RFDETR: The max number of detections "
                 "in the box ("
              << num_detections_boxes
              << ") and "
                 "classes ("
              << num_detections_classes
              << ") tensors "
                 "don't match! Did you pass in the correct model?\n";
    return false;
  }

  auto tensor_classes = layer_to_span<float>(*layer_classes);
  auto tensor_boxes = layer_to_span<float>(*layer_boxes);

  auto width = network.width;
  auto height = network.height;

  // We can add at most num_detection_classes, pre-allocate
  detections.reserve(num_detections_classes);

  for (unsigned int i = 0; i < num_detections_classes; ++i) {
    auto classes = view<float>(tensor_classes, i, num_classes);
    auto boxes = view<float>(tensor_boxes, i, Layer::Boxes::Box::SIZE);

    auto detection = parse_detection(boxes, classes, params, width, height);
    if (!detection) {
      continue;
    }

    detections.push_back(*detection);
  }

  return true;
}

// Unused, just having the compiler check the signature
namespace {
[[maybe_unused]] const NvDsInferParseCustomFunc check_deepstream_rfdetr_bbox =
    deepstream_rfdetr_bbox;
}
