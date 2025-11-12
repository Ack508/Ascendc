/**
 * @file add_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

constexpr int32_t M = 64;
constexpr int32_t K = 32;
constexpr int32_t N = 16;
constexpr int32_t E = 4;

constexpr int32_t xTOTAL_LENGTH = M * K;                            // total length of data
constexpr int32_t wTOTAL_LENGTH = E * K * N;
constexpr int32_t x_scaleTOTAL_LENGTH = M;
constexpr int32_t w_scaleTOTAL_LENGTH = E * N;
constexpr int32_t group_listTOTAL_LENGTH = E;
constexpr int32_t yTOTAL_LENGTH = M * N;

constexpr int32_t USE_CORE_NUM = 1;                                   // num of core used
constexpr int32_t xBLOCK_LENGTH = xTOTAL_LENGTH / USE_CORE_NUM;         // length computed of each core
constexpr int32_t wBLOCK_LENGTH = wTOTAL_LENGTH / USE_CORE_NUM;
constexpr int32_t x_scaleBLOCK_LENGTH = x_scaleTOTAL_LENGTH / USE_CORE_NUM;
constexpr int32_t w_scaleBLOCK_LENGTH = w_scaleTOTAL_LENGTH / USE_CORE_NUM;
constexpr int32_t group_listBLOCK_LENGTH = group_listTOTAL_LENGTH / USE_CORE_NUM;
constexpr int32_t yBLOCK_LENGTH = yTOTAL_LENGTH / USE_CORE_NUM;

constexpr int32_t TILE_NUM = 1;                                       // split data into 8 tiles for each core
constexpr int32_t BUFFER_NUM = 1;                                     // tensor num for each queue
constexpr int32_t xTILE_LENGTH = xBLOCK_LENGTH / TILE_NUM / BUFFER_NUM; // separate to 2 parts, due to double buffer
constexpr int32_t wTILE_LENGTH = wBLOCK_LENGTH / TILE_NUM / BUFFER_NUM;
constexpr int32_t x_scaleTILE_LENGTH = x_scaleBLOCK_LENGTH / TILE_NUM / BUFFER_NUM;
constexpr int32_t w_scaleTILE_LENGTH = w_scaleBLOCK_LENGTH / TILE_NUM / BUFFER_NUM;
constexpr int32_t group_listTILE_LENGTH = group_listBLOCK_LENGTH / TILE_NUM / BUFFER_NUM;
constexpr int32_t yTILE_LENGTH = yBLOCK_LENGTH / TILE_NUM / BUFFER_NUM;


__aicore__ inline void CopyTiling(TCubeTiling *tiling, uint64_t &localMemSize, GM_ADDR tilingGM)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    localMemSize = *reinterpret_cast<__gm__ uint64_t *>(tilingGM + sizeof(TCubeTiling));
    return;
}

class W4a4_gmm {
public:
    __aicore__ inline W4a4_gmm() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w, GM_ADDR x_scale, GM_ADDR w_scale, GM_ADDR group_list, GM_ADDR y)
    {
        //xGm.SetGlobalBuffer((__gm__ int8_t *)x + xBLOCK_LENGTH * AscendC::GetBlockIdx(), xBLOCK_LENGTH);
        xGm.SetGlobalBuffer((__gm__ int8_t *)x + xBLOCK_LENGTH * 0, xBLOCK_LENGTH);
        wGm.SetGlobalBuffer((__gm__ int8_t *)w + wBLOCK_LENGTH * 0, wBLOCK_LENGTH);
        x_scaleGm.SetGlobalBuffer((__gm__ float *)x_scale + x_scaleBLOCK_LENGTH * 0, x_scaleBLOCK_LENGTH);
        w_scaleGm.SetGlobalBuffer((__gm__ float *)w_scale + w_scaleBLOCK_LENGTH * 0, w_scaleBLOCK_LENGTH);
        group_listGm.SetGlobalBuffer((__gm__ int64_t *)group_list + group_listBLOCK_LENGTH * 0, group_listBLOCK_LENGTH);
        yGm.SetGlobalBuffer((__gm__ half *)y + yBLOCK_LENGTH * 0, yBLOCK_LENGTH);

        pipe.InitBuffer(inQueuex, BUFFER_NUM, xTILE_LENGTH * sizeof(int8_t));
        pipe.InitBuffer(inQueuew, BUFFER_NUM, wTILE_LENGTH * sizeof(int8_t));
        pipe.InitBuffer(inQueuex_scale, BUFFER_NUM, x_scaleTILE_LENGTH * sizeof(float));
        pipe.InitBuffer(inQueuew_scale, BUFFER_NUM, w_scaleTILE_LENGTH * sizeof(float));
        pipe.InitBuffer(inQueuegroup_list, BUFFER_NUM, group_listTILE_LENGTH * sizeof(int64_t));
        pipe.InitBuffer(outQueuey, BUFFER_NUM, yTILE_LENGTH * sizeof(uint16_t));
        pipe.InitBuffer(outQueuex, BUFFER_NUM, xTILE_LENGTH * sizeof(int8_t));
        pipe.InitBuffer(outQueuew, BUFFER_NUM, wTILE_LENGTH * sizeof(int8_t));
        pipe.InitBuffer(inQueuematmul, BUFFER_NUM, M * N * sizeof(int32_t));
        pipe.InitBuffer(outQueuew_scale, BUFFER_NUM, w_scaleTILE_LENGTH * sizeof(float));
        pipe.InitBuffer(outQueuex_scale, BUFFER_NUM, x_scaleTILE_LENGTH * sizeof(float));
        pipe.InitBuffer(inQueuefactor, BUFFER_NUM, M*N * sizeof(float));

        pipe.InitBuffer(X_scalebuf,M*sizeof(float));
        pipe.InitBuffer(W_scalebuf,E*N*sizeof(float));
        pipe.InitBuffer(Xbuf, K * M * sizeof(int8_t));
        pipe.InitBuffer(Wbuf, E * K * N * sizeof(int8_t));
        pipe.InitBuffer(Matmulbuf, M * N * sizeof(int32_t));
        pipe.InitBuffer(Factorbuf, M * N * sizeof(float));
        pipe.InitBuffer(Matmulfloatbuf, M * N * sizeof(float));
        pipe.InitBuffer(Yfloatbuf, yTILE_LENGTH * sizeof(float));
        pipe.InitBuffer(Matmulhalfbuf, M * N * sizeof(half));
        //pipe.InitBuffer(X_scale2buf, 2*M*sizeof(float));
    }
    __aicore__ inline void Process(GM_ADDR workspace, GM_ADDR tilingGm1, GM_ADDR tilingGm2)
    {
        CopyIn();
        Compute(workspace, tilingGm1, tilingGm2);
        CopyOut();
        /*
        int32_t loopCount = TILE_NUM * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        */
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<int8_t> xLocal = inQueuex.AllocTensor<int8_t>();
        AscendC::LocalTensor<int8_t> wLocal = inQueuew.AllocTensor<int8_t>();
        AscendC::LocalTensor<float> x_scaleLocal = inQueuex_scale.AllocTensor<float>();
        AscendC::LocalTensor<float> w_scaleLocal = inQueuew_scale.AllocTensor<float>();
        AscendC::LocalTensor<int64_t> group_listLocal = inQueuegroup_list.AllocTensor<int64_t>();

        AscendC::DataCopy(xLocal, xGm, xTILE_LENGTH);
        AscendC::DataCopy(wLocal, wGm, wTILE_LENGTH);
        AscendC::DataCopy(x_scaleLocal, x_scaleGm, x_scaleTILE_LENGTH);
        AscendC::DataCopy(w_scaleLocal, w_scaleGm, w_scaleTILE_LENGTH);
        AscendC::DataCopy(group_listLocal, group_listGm, group_listTILE_LENGTH);

        inQueuex.EnQue(xLocal);
        inQueuew.EnQue(wLocal);
        inQueuex_scale.EnQue(x_scaleLocal);
        inQueuew_scale.EnQue(w_scaleLocal);
        inQueuegroup_list.EnQue(group_listLocal);
        printf("%d: copyin finish\n",AscendC::GetBlockIdx());
    }
    __aicore__ inline void Compute(GM_ADDR workspace, GM_ADDR tilingGm1, GM_ADDR tilingGm2)
    {
        AscendC::LocalTensor<int8_t> xLocal = inQueuex.DeQue<int8_t>();
        AscendC::LocalTensor<int8_t> wLocal = inQueuew.DeQue<int8_t>();
        AscendC::LocalTensor<float> x_scaleLocal = inQueuex_scale.DeQue<float>();
        AscendC::LocalTensor<float> w_scaleLocal = inQueuew_scale.DeQue<float>();
        AscendC::LocalTensor<int64_t> group_listLocal = inQueuegroup_list.DeQue<int64_t>();
        AscendC::LocalTensor<half> yLocal = outQueuey.AllocTensor<half>();
        AscendC::LocalTensor<int8_t> xx=outQueuex.AllocTensor<int8_t>();
        AscendC::LocalTensor<int8_t> ww=outQueuew.AllocTensor<int8_t>();
        AscendC::LocalTensor<int32_t> matm=inQueuematmul.AllocTensor<int32_t>();

        // test
        /*
        printf("%d\n",group_listLocal.GetSize());
        printf("%d\n",xLocal.GetSize());
        printf("%d\n",wLocal.GetSize());
        printf("%d\n",x_scaleLocal.GetSize());
        printf("%d\n",w_scaleLocal.GetSize());

        group_listLocal.Print();
        */
        //wLocal.Print();
        //开始计算 yLocal

        AscendC::LocalTensor<int8_t> w_temp = Wbuf.AllocTensor<int8_t>(); 
        AscendC::LocalTensor<int8_t> x_temp = Xbuf.AllocTensor<int8_t>();
        AscendC::LocalTensor<float> w_scale_temp = W_scalebuf.AllocTensor<float>();
        AscendC::LocalTensor<float> x_scale_temp = X_scalebuf.AllocTensor<float>();
        AscendC::LocalTensor<float> w_scal = outQueuew_scale.AllocTensor<float>();
        AscendC::LocalTensor<float> x_scal = outQueuex_scale.AllocTensor<float>();
        AscendC::LocalTensor<float> fact = inQueuefactor.AllocTensor<float>();

        //AscendC::LocalTensor<float> x_scale2_temp = X_scale2buf.AllocTensor<float>();

        AscendC::DataCopy(w_temp,wLocal,E*K*N);
        AscendC::DataCopy(x_temp,xLocal,M*K);
        AscendC::DataCopy(xx,x_temp,M*K);
        AscendC::DataCopy(ww,w_temp,E*K*N);

        AscendC::DataCopy(w_scale_temp,w_scaleLocal,E*N);
        AscendC::DataCopy(w_scal,w_scale_temp,E*N);
        AscendC::DataCopy(x_scale_temp,x_scaleLocal,M);
        AscendC::DataCopy(x_scal,x_scale_temp,M);

        /*AscendC::DataCopy(w_temp,wLocal,K*N);
        wLocal.Print();
        w_temp.Print();*/
        //wLocal.Print();
        //printf("start\n");

        AscendC::LocalTensor<int32_t> matmul = Matmulbuf.AllocTensor<int32_t>();
        AscendC::LocalTensor<half> matmulhalf = Matmulhalfbuf.AllocTensor<half>();
        AscendC::LocalTensor<float> matmulfloat = Matmulfloatbuf.AllocTensor<float>();
        AscendC::LocalTensor<float> factor = Factorbuf.AllocTensor<float>();
        AscendC::LocalTensor<float> yfloat = Yfloatbuf.AllocTensor<float>();

        // 矩阵乘初始化
        TCubeTiling tiling1,tiling2;
        uint64_t localMemSize1 = 0; //实则没用
        uint64_t localMemSize2 = 0;
        CopyTiling(&tiling1, localMemSize1, tilingGm1);
        CopyTiling(&tiling2, localMemSize2, tilingGm2);

        Matmul<MatmulType<AscendC::TPosition::VECOUT, CubeFormat::ND, int8_t>,
               MatmulType<AscendC::TPosition::VECOUT, CubeFormat::ND, int8_t>,
               MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, int32_t>> mm1;

        Matmul<MatmulType<AscendC::TPosition::VECOUT, CubeFormat::ND, float>,
               MatmulType<AscendC::TPosition::VECOUT, CubeFormat::ND, float>,
               MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, float>> mm2;

        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm1, &tiling1, mm2, &tiling2);
        //REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm2, &tiling2);
        /*mm1.Init(&tiling1);
        mm2.Init(&tiling2);*/

        int32_t line=0;
        int32_t thislines;
        for(int i=0;i<E;i++)// 通过循环得到两个M*N的矩阵：matmul(int32_t)和factor(float)
        {
            thislines=group_listLocal(i);
            //AscendC::DataCopy(w_temp,wLocal[i*K*N],K*N); //本次专家对应的w_temp矩阵
            // todo: 把本次thislines对应的行的子矩阵，与w_temp相乘存到matmul，M*K*N
            mm1.SetOrgShape(thislines, N, K);
            mm1.SetTensorA(xx[line*K]);
            mm1.SetTensorB(ww[i*K*N]);
            //mm1.IterateAll(matm[line*N]);
            while (mm1.Iterate()) {   
                mm1.GetTensorC(matm[line*N]); 
            }
            // todo: 计算factor缩放矩阵，M*1*N
            // 放到另一个循环做了

            // 最后统一转换
            //AscendC::Cast(matmulfloat,matmul,AscendC::RoundMode::CAST_NONE,thislines*N);
            // 把matmulfloat中的结果与factor矩阵进行对应位置相乘，存到yfloat的对应位置
            // 思考：这一步可以放到最后一起乘
            // Mul(yfloat[line*N],matmulfloat,factor,thislines*N);
            
            line=line+thislines;
        }
        mm1.End();
        AscendC::DataCopy(matmul,matm,M*N);
        //matmul.Print();

        line=0;
        mm2.SetOrgShape(M, N, 1);// always
        mm2.SetTensorA(x_scal);// always
        for(int i=0;i<E;i++)// 通过循环得到factor(float)
        {
            thislines=group_listLocal(i);
            // todo: 计算factor缩放矩阵，M*1*N
            // 地址对齐
            //mm2.SetOrgShape(M, N, 1);
            //AscendC::DataCopy(x_scal,x_scale_temp[line*1],M);//一次搬M个数
            //mm2.SetTensorA(x_scal);
            //mm2.SetTensorA(x_scal[line*1]);
            mm2.SetTensorB(w_scal[i*1*N]);
            //mm2.IterateAll(fact[line*N]);
            while (mm2.Iterate()) {   
                mm2.GetTensorC(fact); 
            }
            AscendC::DataCopy(factor[line*N],fact[line*N],thislines*N);
            line=line+thislines;
        }
        mm2.End();
        //AscendC::DataCopy(factor,fact,M*N);
        //factor.Print();
        // matmul: int32_t to float 类型转换
        AscendC::Cast(matmulfloat,matmul,AscendC::RoundMode::CAST_NONE,M*N);
        //AscendC::Cast(matmulfloat,matmulhalf,AscendC::RoundMode::CAST_NONE,M*N);

        //Mul(yfloat,matmulfloat,factor);
        
        yfloat=matmulfloat*factor;

        // y: float to half 类型转换
        AscendC::Cast(yLocal,yfloat,AscendC::RoundMode::CAST_NONE,M*N);
        // 结果tensor进入outQueue
        outQueuey.EnQue<half>(yLocal);
        flag=true;
        // 释放输入内存
        inQueuex.FreeTensor(xLocal);
        inQueuew.FreeTensor(wLocal);
        inQueuex_scale.FreeTensor(x_scaleLocal);
        inQueuew_scale.FreeTensor(w_scaleLocal);
        inQueuegroup_list.FreeTensor(group_listLocal);

        inQueuefactor.FreeTensor(fact);
        inQueuematmul.FreeTensor(matm);
        outQueuex.FreeTensor(xx);
        outQueuew.FreeTensor(ww);
        outQueuew_scale.FreeTensor(w_scal);
        outQueuex_scale.FreeTensor(x_scal);
        printf("%d: compute finish\n",AscendC::GetBlockIdx());
    }
    __aicore__ inline void CopyOut()
    {
        if(flag)
        {
            AscendC::LocalTensor<half> yLocal = outQueuey.DeQue<half>();
            AscendC::DataCopy(yGm, yLocal, yTILE_LENGTH);
            outQueuey.FreeTensor(yLocal);
            flag=false;
            printf("%d: copyout finish\n",AscendC::GetBlockIdx());
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueuex, inQueuew, inQueuex_scale, inQueuew_scale, inQueuegroup_list, inQueuematmul, inQueuefactor;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueuey, outQueuex, outQueuew, outQueuex_scale, outQueuew_scale;
    AscendC::TBuf<AscendC::TPosition::VECCALC> Wbuf, Matmulbuf, Factorbuf, Matmulfloatbuf, Yfloatbuf, Matmulhalfbuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> Xbuf, X_scalebuf, W_scalebuf;

    AscendC::GlobalTensor<int8_t> xGm;
    AscendC::GlobalTensor<int8_t> wGm;
    AscendC::GlobalTensor<float> x_scaleGm;
    AscendC::GlobalTensor<float> w_scaleGm;
    AscendC::GlobalTensor<int64_t> group_listGm;
    AscendC::GlobalTensor<half> yGm;

    bool flag=false;// 队列里是否有内容
};

extern "C" __global__ __aicore__ void w4a4_gmm_custom(GM_ADDR x, GM_ADDR w, GM_ADDR x_scale, GM_ADDR w_scale, GM_ADDR group_list, GM_ADDR y
, GM_ADDR workspace, GM_ADDR tilingGm1, GM_ADDR tilingGm2)
{
    W4a4_gmm op;
    op.Init(x, w, x_scale, w_scale, group_list, y);
    op.Process(workspace, tilingGm1, tilingGm2);
}

