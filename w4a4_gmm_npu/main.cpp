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
extern "C" void w4a4_gmm_custom(uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *);
#endif
extern void GenerateTiling1(const char *socVersion, uint8_t *tilingBuf);

int32_t main(int32_t argc, char *argv[])
{
    const char *socVersion = SOC_VERSION;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);

    //M,K,N,E大小
    int32_t M = 64;
    int32_t K = 32;
    int32_t N = 64;
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

#ifdef CUSTOM_ASCEND310P
    uint32_t blockDim = 2;
#else
    uint32_t blockDim = 1;
#endif

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(xFileSize);
    uint8_t *w = (uint8_t *)AscendC::GmAlloc(wFileSize);
    uint8_t *x_scale = (uint8_t *)AscendC::GmAlloc(x_scaleFileSize);
    uint8_t *w_scale = (uint8_t *)AscendC::GmAlloc(w_scaleFileSize);
    uint8_t *group_list = (uint8_t *)AscendC::GmAlloc(group_listFileSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(yFileSize);

    uint8_t *tiling1 = (uint8_t *)AscendC::GmAlloc(tilingFileSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);

    ReadFile("./input/x.bin", xFileSize, x, xFileSize);
    ReadFile("./input/w.bin", wFileSize, w, wFileSize);
    ReadFile("./input/x_scale.bin", x_scaleFileSize, x_scale, x_scaleFileSize);
    ReadFile("./input/w_scale.bin", w_scaleFileSize, w_scale, w_scaleFileSize);
    ReadFile("./input/group_list.bin", group_listFileSize, group_list, group_listFileSize);

    memcpy_s(tiling1, tilingFileSize, tilingBuf1, tilingFileSize);
    //AscendC::SetKernelMode(KernelMode::AIC_MODE);
    ICPU_RUN_KF(w4a4_gmm_custom, blockDim, x, w, x_scale, w_scale, group_list, y, workspace, tiling1);

    WriteFile("./output/output.bin", y, yFileSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)w);
    AscendC::GmFree((void *)x_scale);
    AscendC::GmFree((void *)w_scale);
    AscendC::GmFree((void *)group_list);
    AscendC::GmFree((void *)y);

    AscendC::GmFree((void *)tiling1);
    AscendC::GmFree((void *)workspace);
#else // NPU侧调用
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *inputHost_x;
    uint8_t *inputDevice_x;
    CHECK_ACL(aclrtMallocHost((void **)(&inputHost_x), xFileSize));
    CHECK_ACL(aclrtMalloc((void **)&inputDevice_x, xFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x.bin", xFileSize, inputHost_x, xFileSize);
    CHECK_ACL(aclrtMemcpy(inputDevice_x, xFileSize, inputHost_x, xFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *inputHost_w;
    uint8_t *inputDevice_w;
    CHECK_ACL(aclrtMallocHost((void **)(&inputHost_w), wFileSize));
    CHECK_ACL(aclrtMalloc((void **)&inputDevice_w, wFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/w.bin", wFileSize, inputHost_w, wFileSize);
    CHECK_ACL(aclrtMemcpy(inputDevice_w, wFileSize, inputHost_w, wFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *inputHost_x_scale;
    uint8_t *inputDevice_x_scale;
    CHECK_ACL(aclrtMallocHost((void **)(&inputHost_x_scale), x_scaleFileSize));
    CHECK_ACL(aclrtMalloc((void **)&inputDevice_x_scale, x_scaleFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x_scale.bin", x_scaleFileSize, inputHost_x_scale, x_scaleFileSize);
    CHECK_ACL(aclrtMemcpy(inputDevice_x_scale, x_scaleFileSize, inputHost_x_scale, x_scaleFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *inputHost_w_scale;
    uint8_t *inputDevice_w_scale;
    CHECK_ACL(aclrtMallocHost((void **)(&inputHost_w_scale), w_scaleFileSize));
    CHECK_ACL(aclrtMalloc((void **)&inputDevice_w_scale, w_scaleFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/w_scale.bin", w_scaleFileSize, inputHost_w_scale, w_scaleFileSize);
    CHECK_ACL(aclrtMemcpy(inputDevice_w_scale, w_scaleFileSize, inputHost_w_scale, w_scaleFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *inputHost_group_list;
    uint8_t *inputDevice_group_list;
    CHECK_ACL(aclrtMallocHost((void **)(&inputHost_group_list), group_listFileSize));
    CHECK_ACL(aclrtMalloc((void **)&inputDevice_group_list, group_listFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/group_list.bin", group_listFileSize, inputHost_group_list, group_listFileSize);
    CHECK_ACL(aclrtMemcpy(inputDevice_group_list, group_listFileSize, inputHost_group_list, group_listFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *outputHost_y;
    uint8_t *outputDevice_y;
    CHECK_ACL(aclrtMallocHost((void **)(&outputHost_y), yFileSize));
    CHECK_ACL(aclrtMalloc((void **)&outputDevice_y, yFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *tilingHost1;
    uint8_t *tilingDevice1;
    CHECK_ACL(aclrtMallocHost((void **)(&tilingHost1), tilingFileSize));
    CHECK_ACL(aclrtMalloc((void **)&tilingDevice1, tilingFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMemcpy(tilingHost1, tilingFileSize, tilingBuf1, tilingFileSize, ACL_MEMCPY_HOST_TO_HOST));
    CHECK_ACL(aclrtMemcpy(tilingDevice1, tilingFileSize, tilingHost1, tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE));


    uint8_t *workspaceDevice;
    CHECK_ACL(aclrtMalloc((void **)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));

    //ICPU_RUN_KF(w4a4_gmm_custom, blockDim, x, w, x_scale, w_scale, group_list, y, workspace, tiling1, tiling2);
    ACLRT_LAUNCH_KERNEL(w4a4_gmm_custom)
    (blockDim, stream, inputDevice_x, inputDevice_w, inputDevice_x_scale, inputDevice_w_scale, inputDevice_group_list,
        outputDevice_y, workspaceDevice, tilingDevice1);

    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtFree(inputDevice_x));
    CHECK_ACL(aclrtFreeHost(inputHost_x));
    CHECK_ACL(aclrtFree(inputDevice_w));
    CHECK_ACL(aclrtFreeHost(inputHost_w));
    CHECK_ACL(aclrtFree(inputDevice_x_scale));
    CHECK_ACL(aclrtFreeHost(inputHost_x_scale));
    CHECK_ACL(aclrtFree(inputDevice_w_scale));
    CHECK_ACL(aclrtFreeHost(inputHost_w_scale));
    CHECK_ACL(aclrtFree(inputDevice_group_list));
    CHECK_ACL(aclrtFreeHost(inputHost_group_list));

    CHECK_ACL(aclrtMemcpy(outputHost_y, yFileSize, outputDevice_y, yFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output.bin", outputHost_y, yFileSize);
    CHECK_ACL(aclrtFree(outputDevice_y));
    CHECK_ACL(aclrtFreeHost(outputHost_y));
    
    CHECK_ACL(aclrtFree(tilingDevice1));
    CHECK_ACL(aclrtFreeHost(tilingHost1));
    CHECK_ACL(aclrtFree(workspaceDevice));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    free(tilingBuf1);
    return 0;
}