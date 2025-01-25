#!/bin/bash

exp_root=./models
eval_tasks=("linear_regression")
eval_dims=("5-20")
eval_date=("test")

containsElement () {
    local e
    for e in "${@:2}"; do [[ "$e" == "$1" ]] && return 0; done
    return 1
}

for t in $(ls $exp_root); do
    if containsElement "$t" "${eval_tasks[@]}"; then
        for d in $(ls $exp_root/$t); do
            if containsElement "$d" "dim_${eval_dims[@]}"; then
                for e in $(ls $exp_root/$t/$d); do
                    if containsElement "$e" "${eval_date[@]}"; then
                        for f in $(ls $exp_root/$t/$d/$e); do
                            echo "Evaluating task $t for $f"
                            python3 eval_attn.py $exp_root/$t/$d/$e/$f
                        done
                    else
                        echo "Skipping date $e for task $t with dim $d"
                        continue
                    fi
                done
            fi
        done
    else
        echo "Skipping task $t"
        continue
    fi
done
