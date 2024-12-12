#!/bin/bash

set -e

function run_cmd {
    docker_command="docker run -d --rm "
    options="--net=host --shm-size=10.24gb"
    image="stap-train"

    if [ "$gpu" = "gpu" ]
    then
        options="${options} --gpus all"
        image="${image}-gpu"
    fi

    if [ "$user" = "root" ]
        then
        options="${options} --volume="$(pwd)/models/:/root/models/""
        image="${image}/root:v2"
    elif [ "$user" = "user" ]
        then
        options="${options} --volume="$(pwd)/models/:/home/$USER/models/" --user=$USER"
        image="${image}/$USER:v2"
    else
        echo "User mode unknown. Please choose user, root, or leave out for default user"
    fi

    echo "Running docker command: ${docker_command} ${options} ${image} ${CMD}"

    ${docker_command} \
        ${options} \
        ${image} \
        "${CMD}"
}

function eval_planner {
    args=""
    args="${args} --planner-config ${PLANNER_CONFIG}"
    args="${args} --env-config ${ENV_CONFIG}"
    if [ ${#POLICY_CHECKPOINTS[@]} -gt 0 ]; then
        args="${args} --policy-checkpoints ${POLICY_CHECKPOINTS[@]}"
    fi
    if [ ${#SCOD_CHECKPOINTS[@]} -gt 0 ]; then
        args="${args} --scod-checkpoints ${SCOD_CHECKPOINTS[@]}"
    fi
    if [ ! -z "${DYNAMICS_CHECKPOINT}" ]; then
        args="${args} --dynamics-checkpoint ${DYNAMICS_CHECKPOINT}"
    fi
    if [[ ! -z "${LOAD_PATH}" ]]; then
        args="${args} --load-path ${LOAD_PATH}"
    fi
    args="${args} --seed 0"
    args="${args} ${ENV_KWARGS}"
    if [[ $DEBUG -ne 0 ]]; then
        args="${args} --num-eval 1"
        args="${args} --path ${PLANNER_OUTPUT_PATH}_debug"
        args="${args} --verbose 1"
    else
        args="${args} --num-eval 100"
        args="${args} --path ${PLANNER_OUTPUT_PATH}"
        args="${args} --verbose 1"
    fi
    CMD="python scripts/eval/eval_planners.py ${args}"
    run_cmd
}

function run_planners {
    for task in "${TASKS[@]}"; do
        PLANNER_OUTPUT_PATH="${PLANNER_OUTPUT_ROOT}/${task}"
        ENV_CONFIG="${TASK_ROOT}/${task}.yaml"

        for planner in "${PLANNERS[@]}"; do
            PLANNER_CONFIG="${PLANNER_CONFIG_PATH}/${planner}.yaml"

            POLICY_CHECKPOINTS=()
            for primitive in "${PRIMITIVES[@]}"; do
                POLICY_CHECKPOINTS+=("${POLICY_INPUT_PATH}/${primitive}/${CHECKPOINT}.pt")
            done

            SCOD_CHECKPOINTS=()

            if [[ "${planner}" == *_oracle_*dynamics ]]; then
                DYNAMICS_CHECKPOINT=""
            else
                DYNAMICS_CHECKPOINT="${DYNAMICS_INPUT_PATH}/${CHECKPOINT}.pt"
            fi
    
            eval_planner
        done
    done
}

# Evaluation tasks: Uncomment tasks to evaluate.
TASK_ROOT="configs/pybullet/envs/official/sim_domains"
TASKS=(
    "object_arrangement/generated_ablation_trial_0"
    "object_arrangement/generated_ablation_trial_1"
    "object_arrangement/generated_ablation_trial_2"
    "object_arrangement/generated_ablation_trial_3"
    "object_arrangement/generated_ablation_trial_4"
    "object_arrangement/generated_ablation_trial_5"
    "object_arrangement/generated_ablation_trial_6"
    "object_arrangement/generated_ablation_trial_7"
    "object_arrangement/generated_ablation_trial_8"
    "object_arrangement/generated_ablation_trial_9"
    "object_arrangement/generated_ablation_trial_10"
    "object_arrangement/generated_ablation_trial_11"
    "object_arrangement/generated_ablation_trial_12"
    "object_arrangement/generated_ablation_trial_13"
    "object_arrangement/generated_ablation_trial_14"
    "object_arrangement/generated_ablation_trial_15"
    "object_arrangement/generated_ablation_trial_16"
    "object_arrangement/generated_ablation_trial_17"
    "object_arrangement/generated_ablation_trial_18"
    "object_arrangement/generated_ablation_trial_19"
    "object_arrangement/generated_ablation_trial_20"
    "object_arrangement/generated_ablation_trial_21"
    "object_arrangement/generated_ablation_trial_22"
    "object_arrangement/generated_ablation_trial_23"
    "object_arrangement/generated_ablation_trial_24"
    "object_arrangement/generated_ablation_trial_25"
    "object_arrangement/generated_ablation_trial_26"
    "object_arrangement/generated_ablation_trial_27"
    "object_arrangement/generated_ablation_trial_28"
    "object_arrangement/generated_ablation_trial_29"
)

# Planners: Uncomment planners to evaluate.
PLANNER_CONFIG_PATH="configs/pybullet/planners"
PLANNERS=(
    "policy_cem_arrangement_no_custom_fns"
)

# Setup.
user=${1:-user}
gpu=${2:-cpu}

if [ "$user" = "root" ]
    then
    STAP_PATH="/root"
elif [ "$user" = "user" ]
    then
    STAP_PATH="/home/$USER"
else
    echo "User mode unknown. Please choose user, root, or leave out for default user"
fi

input_path="${STAP_PATH}/models"
output_path="${STAP_PATH}/models/eval"

ENV_KWARGS="--gui 0 --closed-loop 1 --use_informed_dynamics 1"

# Evaluate planners.
exp_name="planning"
PLANNER_OUTPUT_ROOT="${output_path}/${exp_name}"
PRIMITIVES=("pick" "place" "static_handover")

CHECKPOINT="final_model"
POLICY_INPUT_PATH="${input_path}/policies_irl"
DYNAMICS_INPUT_PATH="${input_path}/dynamics_irl/pick_place_static_handover_dynamics"
run_planners