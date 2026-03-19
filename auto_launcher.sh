#!/bin/bash
# auto_launcher.sh - 监控R2结束后自动启动R3/长尾任务，并自动接力新完成的ingest实例
# 用法: nohup bash auto_launcher.sh > out/logs/auto_launcher.log 2>&1 &

cd /Users/bytedance/proj/memoRaxis
LOG="out/logs/auto_launcher.log"
echo "[$(date '+%H:%M:%S')] 自动调度器启动" | tee -a $LOG

r3_launched=false

while true; do
    sleep 120  # 每2分钟检查一次

    # ====== 1. R2全部完成后，启动R3和long_range R2/R3 ======
    if ! $r3_launched && ! pgrep -qf "output_suffix r2"; then
        echo "[$(date '+%H:%M:%S')] R2全部完成，启动R3 + long_range R2/R3..." | tee -a $LOG

        # R3 conflict 0-5
        for idx in 0 1 2 3 4 5; do
            nohup .venv/bin/python3 scripts/mem0_MAB/infer.py \
                --task Conflict_Resolution --instance_idx $idx \
                --adaptor R3 --output_suffix r3 --limit -1 \
                > out/logs/infer_conflict_r3_${idx}.log 2>&1 &
            echo "[$(date '+%H:%M:%S')] R3 conflict_${idx}: PID $!" | tee -a $LOG
            sleep 2
        done

        # R3 TTL 1-3
        for idx in 1 2 3; do
            nohup .venv/bin/python3 scripts/mem0_MAB/infer.py \
                --task Test_Time_Learning --instance_idx $idx \
                --adaptor R3 --output_suffix r3 --limit -1 \
                > out/logs/infer_ttl_r3_${idx}.log 2>&1 &
            echo "[$(date '+%H:%M:%S')] R3 TTL_${idx}: PID $!" | tee -a $LOG
        done

        # R2 + R3 long_range 0-109
        nohup .venv/bin/python3 scripts/mem0_MAB/infer.py \
            --task Long_Range_Understanding --instance_idx 0-109 \
            --adaptor R2 --output_suffix r2 --limit -1 \
            > out/logs/infer_long_range_r2.log 2>&1 &
        echo "[$(date '+%H:%M:%S')] R2 long_range: PID $!" | tee -a $LOG

        nohup .venv/bin/python3 scripts/mem0_MAB/infer.py \
            --task Long_Range_Understanding --instance_idx 0-109 \
            --adaptor R3 --output_suffix r3 --limit -1 \
            > out/logs/infer_long_range_r3.log 2>&1 &
        echo "[$(date '+%H:%M:%S')] R3 long_range: PID $!" | tee -a $LOG

        r3_launched=true
    fi

    # ====== 2. 监控acc_ret新完成的ingest实例，自动启动R1+R2+R3 ======
    for idx in 1 13 12 16; do
        result="out/mem0/mem0_acc_ret_results_${idx}.json"
        ckpt="checkpoints/mem0_MAB/accurate_retrieval_${idx}.json"
        running=$(pgrep -f "infer.py.*Accurate_Retrieval.*instance_idx ${idx}" 2>/dev/null)
        if [ ! -f "$result" ] && [ -z "$running" ] && [ -f "$ckpt" ]; then
            done=$(python3 -c "
import json
d=json.load(open('$ckpt'))
print('yes' if d.get('ingested',0) >= d.get('total',1) else 'no')
" 2>/dev/null)
            if [ "$done" = "yes" ]; then
                nohup .venv/bin/python3 scripts/mem0_MAB/infer.py \
                    --task Accurate_Retrieval --instance_idx $idx \
                    --adaptor R1 R2 R3 --limit -1 \
                    > out/logs/infer_acc_ret_all_${idx}.log 2>&1 &
                echo "[$(date '+%H:%M:%S')] 自动接力 acc_ret_${idx} R1+R2+R3: PID $!" | tee -a $LOG
            fi
        fi
    done

    # ====== 3. 检查是否已无任务可跑 ======
    if ! pgrep -qf "mem0_MAB\|mem0g_MAB"; then
        echo "[$(date '+%H:%M:%S')] 所有进程完成，调度器退出" | tee -a $LOG
        break
    fi
done
