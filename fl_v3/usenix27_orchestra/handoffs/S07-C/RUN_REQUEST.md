# S07-C RUN_REQUEST

## 状态

```text
SESSION_ID: S07-C
APPROVAL_STATE: NOT REQUESTED / NOT APPROVED
APPROVED_COMPUTE: none
```

本 session 是 legacy-security cleanup 与 clean-foundation preservation，只运行
login-node-safe 的静态/语法/文件检查。没有请求或使用 O-009；没有提交任何
`sbatch`/`srun`。

## 明确 NOT RUN / FORBIDDEN

- Slurm/GH200、Ray/Flower live simulation、spconv CUDA；
- full cache/trainval materialization 或 scan；
- model step、100/1000-step、tiny-overfit、training/evaluation campaign；
- official metric、ASR、profile、DDP、matrix、seed/rerun/retry；
- attack、defense、Protocol-B split 或任何 S12/S13/S14 工作；
- persistent environment mutation/uninstall；
- upload、merge、push、publication。

若未来要补 dependency-backed focused tests，必须由 owner/S00 另行提供 exact
immutable RUN_REQUEST，包括 worker SHA、diff/source manifest、命令、tests、数据
范围、node/GPU/CPU/time、output、stop conditions 和 approval；本文件不构成授权。
