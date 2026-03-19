# Results Manifest
更新: 2026-03-08 | 标准: 100q(LR=1q,TTL0=200q), error率<5%

---

## simpleMemory (`results/simpleMemory/`)  ← 来自 out/simpleMemory_MAB/results/

| 数据集 | 状态 | 备注 |
|--------|------|------|
| AR inst 0-16 | ✅ R1/R2/R3 全完整 | |
| AR inst 17-21 | ⏳ 需补跑 | checkpoint存在，resume即可 |
| CR inst 0-7 | ✅ R1/R2/R3 全完整 | inst2 R2 有5个err，已收录 |
| LR inst 0-39 | ✅ R1/R2/R3 全完整 | 每inst 1题，正常 |
| TTL inst 0 | ⚠️ R1/R2完整，R3中断 | R3需resume |
| TTL inst 1,3 | ✅ R1完整 | ICL任务，R2/R3跳过 |
| TTL inst 2,4 | ✅ R1完整(100q) | ICL，R2/R3 error率过高不收录 |
| TTL inst 5 | ⏳ 需补跑 | |

## MemGPT (`results/memgpt/`)  ← 命名: `{ds}_inst{NN}_{rx}.json`

### Letta Ingest 状态
| 数据集 | 已 ingest | 需 ingest |
|--------|----------|---------|
| AR (22) | 0,1,2,12-21 (13个) | 3,4,5,6,7,8,9,10,11 |
| CR (8) | 0-7 ✅ 全部 | — |
| LR (40) | 0-7,20-39 (28个) | 8-19 |
| TTL (6) | 0 | 1-5 |

### AR 完成矩阵 (12/66 = 18%)
```
       R1    R2    R3
inst0  ✅    ✅    ❌
inst1  ❌    ❌    ❌   ← ingest存在
inst2  ❌    ❌    ✅   ← ingest存在
inst3  ❌    ❌    ❌   ← 需先ingest
inst4  ✅    ✅    ✅
inst5  ❌    ❌    ❌   ← 需先ingest
inst6-9 ❌   ❌    ❌   ← 需先ingest
inst10 ❌    ✅    ❌   ← ingest存在
inst11 ❌    ✅    ❌   ← ingest存在
inst12 ✅    ❌    ❌
inst13 ❌    ❌    ✅
inst14 ❌    ❌    ❌   ← ingest存在
inst15 ❌    ❌    ❌   ← ingest存在
inst16 ✅    ✅    ❌
inst17-21 ❌  ❌   ❌  ← ingest存在
```

### CR 完成矩阵 (8/24 = 33%)
- R1: 全部 inst 0-7 ✅
- R2: 全部缺失 ❌
- R3: 全部缺失 ❌

### LR 完成矩阵 (0/120 = 0%)
全部缺失，需跑 run_plan.sh PHASE 2

### TTL 完成矩阵
- inst0 (recsys): R1/R2/R3 全缺 ❌
- inst1-5 (ICL): R1: 1,2,3,4,5 ✅；R2/R3 设计上跳过

---

## 运行计划 (`run_plan.sh`)
```
PHASE 0: simpleMemory AR 17-21 补跑 + TTL inst5 + TTL inst0 R3  (~2h)
PHASE 1: MemGPT AR 全补（先ingest 3-11，再infer）              (~多天)
PHASE 2: MemGPT LR（先ingest 8-19，再infer）                  (~多天)
PHASE 3: MemGPT CR R2+R3（最重，7inst×100q）                  (~多天)
PHASE 4: MemGPT TTL inst0 R1/R2/R3                           (~1h)
```

## 已知 Bug
- TTL ICL inst(1-5) 与 R2/R3 adaptor 不兼容 → 只运行 R1
- 原因：ICL template "Only output {label}" 与 R2 decision JSON prompt 冲突
