# Ascendc
## 文件位置
w4a4_gmm 单核版本位于 ```w4a4_gmm``` 文件夹，w4a4_gmm 多核版本位于 ```w4a4_gmm_opt``` 文件夹。

以上只支持 cpu 运行，原因是 npu 运行时不支持同时定义两个 matmul 对象。故采用另一种 broadcast 的方法重新实现了算子，单核和多核版本分别在 ```w4a4_gmm_npu``` 和 ```w4a4_gmm_opt_npu``` 文件夹。
## 运行方式
```bash run.sh -r cpu -v Ascend910B3 # cpu运行```

```bash run.sh -r npu -v Ascend910B3 # npu运行，注意要先加载环境```

