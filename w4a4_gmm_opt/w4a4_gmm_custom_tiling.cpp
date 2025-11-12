/**
 * @file matmul_custom_tiling.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
using namespace matmul_tiling;
using namespace std;

/**
  * @brief  Generate matmul tiling.
  * @param  socVersion: Platform socversion.
  * @param  tilingBuf data buffer.
  */
void GenerateTiling1(const char *socVersion, uint8_t *tilingBuf)
{
    constexpr int32_t M = 64;
    constexpr int32_t N = 64;
    constexpr int32_t K = 32;

    TPosition leftPosition = TPosition::VECOUT;
    CubeFormat leftFormat = CubeFormat::ND;
    DataType leftDtype = DataType::DT_INT8;
    bool isTransA = false;

    TPosition rightPosition = TPosition::VECOUT;
    CubeFormat rightFormat = CubeFormat::ND;
    DataType rightDtype = DataType::DT_INT8;
    bool isTransB = false;

    TPosition resultPosition = TPosition::VECIN;
    CubeFormat resultFormat = CubeFormat::ND;
    DataType resultDtype = DataType::DT_INT32;

    bool isBias = false;

    /*constexpr int32_t SINGLECORE_M = 32;
    constexpr int32_t SINGLECORE_N = 8;*/

    optiling::TCubeTiling tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    MultiCoreMatmulTiling tilingApi(*ascendcPlatform);

    tilingApi.SetDim(8); 
    //tilingApi.SetDim(ascendcPlatform->GetCoreNumAiv()); 
    // Set the number of cores that participate in multi-core computaion is 48.
    tilingApi.SetAType(leftPosition, leftFormat, leftDtype, isTransA);
    tilingApi.SetBType(rightPosition, rightFormat, rightDtype, isTransB);
    tilingApi.SetCType(resultPosition, resultFormat, resultDtype);

    tilingApi.SetOrgShape(M, N, K);
    tilingApi.SetShape(M, N, K);
    tilingApi.SetSingleShape(M, N, K);
    /*
    if (ascendcPlatform->GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
        tilingApi.SetSingleShape(SINGLECORE_M, SINGLECORE_N, -1);  // Set the fixed singleCoreM=512, singleCoreN=512.
        int32_t mBlockNum = M / SINGLECORE_M;
        int32_t nBlockNum = N / SINGLECORE_N;
        tilingApi.SetDim(mBlockNum * nBlockNum);
    }*/
    tilingApi.SetBias(isBias);
    tilingApi.SetBufferSpace(-1, -1, -1);

    int64_t res = tilingApi.GetTiling(tilingData); // Get matmul tiling data.
    if (res == -1) {
        std::cout << "gen tiling failed" << std::endl;
    }
    uint32_t tcubeTilingSize = tilingData.GetDataSize();
    tilingData.SaveToBuffer(tilingBuf, tcubeTilingSize);

    uint64_t localMemSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, localMemSize);
    *reinterpret_cast<uint64_t *>(tilingBuf + tcubeTilingSize) = localMemSize;
    return;
}

void GenerateTiling2(const char *socVersion, uint8_t *tilingBuf)
{
    constexpr int32_t M = 64;
    constexpr int32_t N = 64;
    constexpr int32_t K = 1;

    TPosition leftPosition = TPosition::VECOUT;
    CubeFormat leftFormat = CubeFormat::ND;
    DataType leftDtype = DataType::DT_FLOAT;
    bool isTransA = false;

    TPosition rightPosition = TPosition::VECOUT;
    CubeFormat rightFormat = CubeFormat::ND;
    DataType rightDtype = DataType::DT_FLOAT;
    bool isTransB = false;

    TPosition resultPosition = TPosition::VECIN;
    CubeFormat resultFormat = CubeFormat::ND;
    DataType resultDtype = DataType::DT_FLOAT;

    bool isBias = false;

    /*constexpr int32_t SINGLECORE_M = 32;
    constexpr int32_t SINGLECORE_N = 8;*/

    optiling::TCubeTiling tilingData;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    MultiCoreMatmulTiling tilingApi(*ascendcPlatform);

    tilingApi.SetDim(8); 
    //tilingApi.SetDim(ascendcPlatform->GetCoreNumAiv());
    // Set the number of cores that participate in multi-core computaion is 48.
    tilingApi.SetAType(leftPosition, leftFormat, leftDtype, isTransA);
    tilingApi.SetBType(rightPosition, rightFormat, rightDtype, isTransB);
    tilingApi.SetCType(resultPosition, resultFormat, resultDtype);

    tilingApi.SetOrgShape(M, N, K);
    tilingApi.SetShape(M, N, K);
    tilingApi.SetSingleShape(M, N, K);
    /*
    if (ascendcPlatform->GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P) {
        tilingApi.SetSingleShape(SINGLECORE_M, SINGLECORE_N, -1);  // Set the fixed singleCoreM=512, singleCoreN=512.
        int32_t mBlockNum = M / SINGLECORE_M;
        int32_t nBlockNum = N / SINGLECORE_N;
        tilingApi.SetDim(mBlockNum * nBlockNum);
    }*/
    tilingApi.SetBias(isBias);
    tilingApi.SetBufferSpace(-1, -1, -1);

    int64_t res = tilingApi.GetTiling(tilingData); // Get matmul tiling data.
    if (res == -1) {
        std::cout << "gen tiling failed" << std::endl;
    }
    uint32_t tcubeTilingSize = tilingData.GetDataSize();
    tilingData.SaveToBuffer(tilingBuf, tcubeTilingSize);

    uint64_t localMemSize;
    ascendcPlatform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, localMemSize);
    *reinterpret_cast<uint64_t *>(tilingBuf + tcubeTilingSize) = localMemSize;
    return;
}