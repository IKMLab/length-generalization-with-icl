#!/bin/bash

configs_root=./configs/tasks
run_task=("linear_regression")
run_pe=("alibi" "dynamic_yarn" "fire" "nope")

containsElement () {
    local e
    for e in "${@:2}"; do [[ "$e" == "$1" ]] && return 0; done
    return 1
}

for t in $(ls $configs_root); do
    if containsElement "$t" "${run_task[@]}"; then
        for f in $(ls $configs_root/$t); do
            if containsElement "${f%.*}" "${run_pe[@]}"; then
                echo "Training task $t with strategy $f"
                python3 train.py $configs_root/$t/$f
            else
                echo "Skipping strategy $f for task $t"
                continue
            fi
        done
    else
        echo "Skipping task $t"
        continue
    fi
done