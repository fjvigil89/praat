#!/bin/bash

# C="0.015625 0.03125 0.0625 0.125 0.25 0.5 1 2 4 8 16 32"
# G="0.000244140625 0.00048828125 0.0009765625 0.001953125 0.00390625 0.0078125 0.015625 0.03125 0.0625 0.125 0.25"
# D="1 2 3"
# Ker="poly rbf"
 
C="0.015625"
G="0.000244140625"
D="1"
Ker="poly"
 
for k in $Ker; do
    for c in $C; do   
        case $k in
            "poly")
                for d in $D; do
                    LOGDIR=log
                    mkdir -p $LOGDIR
                    TASK_FILE=$LOGDIR/task_${k}_${c}_${d}                ​
                    echo "
                        executable =  /extra/scratch03/fjvigil/baseline/src/model/run_cluster.py
                        arguments = \"${k} ${c} ${d}\"
                        output = ${TASK_FILE}.log
                        error = ${TASK_FILE}.err
                        log = ${TASK_FILE}.clog
                        queue
                    " > $TASK_FILE                ​
                    # condor_submit.sh -c 1 -r 3900 --config $TASK_FILE
                done    
                ;;
            "rbf")
                for g in $G; do
                    LOGDIR=log
                    mkdir -p $LOGDIR
                    TASK_FILE=$LOGDIR/task_${k}_${c}_${g}                ​
                    echo "
                        executable =  /extra/scratch03/fjvigil/baseline/src/model/cluster.py
                        arguments = \"${k} ${c} ${d}\"
                        output = ${TASK_FILE}.log
                        error = ${TASK_FILE}.err
                        log = ${TASK_FILE}.clog
                        queue
                    " > $TASK_FILE                ​
                    # condor_submit.sh -c 1 -r 3900 --config $TASK_FILE
                done    
                ;;            
        esac 
    done    
done
