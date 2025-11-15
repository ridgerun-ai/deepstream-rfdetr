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

#include <nvdsinfer_custom_impl.h>

extern "C"
bool deepstream_rfdetr_bbox (const std::vector<NvDsInferLayerInfo> &layers,
			     const NvDsInferNetworkInfo &network,
			     const NvDsInferParseDetectionParams &params,
			     std::vector<NvDsInferObjectDetectionInfo> &detections)
{
    return true;
}

// Unused, just having the compiler check the signature
[[maybe_unused]] static
NvDsInferParseCustomFunc _deepstream_rfdetr_bbox = deepstream_rfdetr_bbox;
