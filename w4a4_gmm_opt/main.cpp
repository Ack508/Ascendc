/**
 * @file main.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "data_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_w4a4_gmm_custom.h"
#else
#include "tikicpulib.h"
extern "C" void w4a4_gmm_custom(uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);
#endif
extern void GenerateTiling1(const char *socVersion, uint8_t *tilingBuf);
extern void GenerateTiling2(const char *socVersion, uint8_t *tilingBuf);

int32_t main(int32_t argc, char *argv[])
{
    const char *socVersion = SOC_VERSION;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);

    //M,K,N,E大小
    int32_t M = 64;
    int32_t K = 32;
    int32_t N = 64*8; //N分到8个核计算
    int32_t E = 4;

    size_t xFileSize = M * K * sizeof(int8_t); //python没有int4，无法制造int4的输入，用int8替代
    size_t wFileSize = E * K * N * sizeof(int8_t);
    size_t x_scaleFileSize = M * sizeof(float);
    size_t w_scaleFileSize = E * N * sizeof(float);
    size_t group_listFileSize = E * sizeof(int64_t);
    size_t yFileSize = M * N * sizeof(uint16_t);

    size_t tilingFileSize = sizeof(TCubeTiling) + sizeof(uint64_t);
    //size_t tilingFileSize = sizeof(TCubeTiling) ;
    size_t userWorkspaceSize = 0;
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    size_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;
    uint8_t *tilingBuf1 = (uint8_t *)malloc(tilingFileSize);
    GenerateTiling1(socVersion, tilingBuf1);
    uint8_t *tilingBuf2 = (uint8_t *)malloc(tilingFileSize);
    GenerateTiling2(socVersion, tilingBuf2);
    printf("space:%ld",workspaceSize);

#ifdef CUSTOM_ASCEND310P
    uint32_t blockDim = 2;
#else
    uint32_t blockDim = 8;
#endif

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xFileSize);
    uint8_t *w = (uint8_t *)AscendC::GmAlloc(wFileSize);
    uint8_t *x_scale = (uint8_t *)AscendC::GmAlloc(x_scaleFileSize);
    uint8_t *w_scale = (uint8_t *)AscendC::GmAlloc(w_scaleFileSize);
    uint8_t *group_list = (uint8_t *)AscendC::GmAlloc(group_listFileSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(yFileSize);

    uint8_t *tiling1 = (uint8_t *)AscendC::GmAlloc(tilingFileSize);
    uint8_t *tiling2 = (uint8_t *)AscendC::GmAlloc(tilingFileSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize*2);

    ReadFile("./input/x.bin", xFileSize, x, xFileSize);
    ReadFile("./input/w.bin", wFileSize, w, wFileSize);
    ReadFile("./input/x_scale.bin", x_scaleFileSize, x_scale, x_scaleFileSize);
    ReadFile("./input/w_scale.bin", w_scaleFileSize, w_scale, w_scaleFileSize);
    ReadFile("./input/group_list.bin", group_listFileSize, group_list, group_listFileSize);

    memcpy_s(tiling1, tilingFileSize, tilingBuf1, tilingFileSize);
    memcpy_s(tiling2, tilingFileSize, tilingBuf2, tilingFileSize);
    //AscendC::SetKernelMode(KernelMode::AIC_MODE);
    ICPU_RUN_KF(w4a4_gmm_custom, blockDim, x, w, x_scale, w_scale, group_list, y, workspace, tiling1, tiling2);

    WriteFile("./output/output.bin", y, yFileSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)w);
    AscendC::GmFree((void *)x_scale);
    AscendC::GmFree((void *)w_scale);
    AscendC::GmFree((void *)group_list);
    AscendC::GmFree((void *)y);

    AscendC::GmFree((void *)tiling1);
    AscendC::GmFree((void *)tiling2);
    AscendC::GmFree((void *)workspace);
#else // todo: NPU侧调用
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *inputAHost;
    uint8_t *inputADevice;
    CHECK_ACL(aclrtMallocHost((void **)(&inputAHost), aFileSize));
    CHECK_ACL(aclrtMalloc((void **)&inputADevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x1_gm.bin", aFileSize, inputAHost, aFileSize);
    CHECK_ACL(aclrtMemcpy(inputADevice, aFileSize, inputAHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *inputBHost;
    uint8_t *inputBDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&inputBHost), bFileSize));
    CHECK_ACL(aclrtMalloc((void **)&inputBDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x2_gm.bin", bFileSize, inputBHost, bFileSize);
    CHECK_ACL(aclrtMemcpy(inputBDevice, bFileSize, inputBHost, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *outputCHost;
    uint8_t *outputCDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&outputCHost), cFileSize));
    CHECK_ACL(aclrtMalloc((void **)&outputCDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *inputBiasHost;
    uint8_t *inputBiasDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&inputBiasHost), biasFileSize));
    CHECK_ACL(aclrtMalloc((void **)&inputBiasDevice, biasFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/bias.bin", biasFileSize, inputBiasHost, biasFileSize);
    CHECK_ACL(aclrtMemcpy(inputBiasDevice, biasFileSize, inputBiasHost, biasFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *tilingHost;
    uint8_t *tilingDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&tilingHost), tilingFileSize));
    CHECK_ACL(aclrtMalloc((void **)&tilingDevice, tilingFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(tilingHost, tilingFileSize, tilingBuf, tilingFileSize, ACL_MEMCPY_HOST_TO_HOST));
    CHECK_ACL(aclrtMemcpy(tilingDevice, tilingFileSize, tilingHost, tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *workspaceDevice;
    CHECK_ACL(aclrtMalloc((void **)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ACLRT_LAUNCH_KERNEL(matmul_leakyrelu_custom)
    (blockDim, stream, inputADevice, inputBDevice, inputBiasDevice, outputCDevice, workspaceDevice, tilingDevice);

    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtFree(inputADevice));
    CHECK_ACL(aclrtFreeHost(inputAHost));
    CHECK_ACL(aclrtFree(inputBDevice));
    CHECK_ACL(aclrtFreeHost(inputBHost));
    CHECK_ACL(aclrtMemcpy(outputCHost, cFileSize, outputCDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output.bin", outputCHost, cFileSize);
    CHECK_ACL(aclrtFree(outputCDevice));
    CHECK_ACL(aclrtFreeHost(outputCHost));
    CHECK_ACL(aclrtFree(inputBiasDevice));
    CHECK_ACL(aclrtFreeHost(inputBiasHost));
    CHECK_ACL(aclrtFree(tilingDevice));
    CHECK_ACL(aclrtFreeHost(tilingHost));
    CHECK_ACL(aclrtFree(workspaceDevice));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    free(tilingBuf1);
    free(tilingBuf2);
    return 0;
}