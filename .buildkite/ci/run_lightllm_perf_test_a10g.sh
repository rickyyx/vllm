#!/usr/bin/env bash

# The below benchmarks run for LightLLM inference engine

NAME="default"
MODEL="7b"
TP="1"
PROMPT_LEN="512"
GEN_LEN="128"
REQUEST_NUM="10"
NUM_USERS="1"
DOWNLOAD_DIR="/mnt/local_storage"
MAX_NUM_BATCHED_TOKENS="5120"
PORT="8000"

INSTALL_S3_PATH="s3://vllm-ci-opensource-perf"
MODEL_S3_PATH="s3://large-dl-models-mirror/models--meta-llama--Llama-2-${MODEL}-chat-hf/main-safetensors/"
LOCAL_PATH="${DOWNLOAD_DIR}/llama-${MODEL}/"

function install_lightllm {
    # Install LightLLM in Python 3.9 environment
    eval "$(conda shell.bash hook)"
    conda create -n lightllm python=3.9 -y
    conda activate lightllm
    aws s3 cp "${INSTALL_S3_PATH}/llmval.tar.gz" $DOWNLOAD_DIR
    aws s3 cp "${INSTALL_S3_PATH}/lightllm.tar.gz" $DOWNLOAD_DIR
    pip install "${DOWNLOAD_DIR}/llmval.tar.gz"
    pip install "${DOWNLOAD_DIR}/lightllm.tar.gz"
    pip install triton==2.0.0.dev20221202
    pip install transformers --upgrade
    pip install 'fschat[model_worker,webui]'
    pip install hf_transfer
    sudo apt-get install netcat -y
}

function download_weights {
    # Download full Llama weights from S3
    if [ ! -f "${LOCAL_PATH}/download_weights.finished" ]
    then
        echo '========================='
        echo "downloading model weights from ${MODEL_S3_PATH}"
        mkdir -p $LOCAL_PATH
        echo "Downloading weights to ${LOCAL_PATH}"
        # LightLLM cannot load dummy weights
        # downloading all weights from s3
        aws s3 cp $MODEL_S3_PATH $LOCAL_PATH --recursive
        echo "Downloading llmval legacy benchmark script with lightllm grpc support"
        aws s3 cp "${INSTALL_S3_PATH}/benchmark_llmval.py" benchmarks/anyscale/
        touch "${LOCAL_PATH}/download_weights.finished"
        echo '========================='
        echo 'model weights downloaded'
    else
        echo '================================'
        echo "model weights already downloaded in ${LOCAL_PATH}"
    fi
}

function start_lightllm_server {
    # Start LightLLM model HTTP server
    echo '====================='
    echo 'starting model server'
    ulimit -n 65536 && python -m lightllm.server.api_server \
        --port $PORT \
        --model_dir $LOCAL_PATH \
        --tp $TP \
        --max_total_token_num 16000 \
        --tokenizer_mode auto \
        --batch_max_token $MAX_NUM_BATCHED_TOKENS \
        2>&1 &
    
    echo "Waiting lightllm server to launch on ${PORT}..."

    while ! nc -z localhost ${PORT}; do   
        sleep 0.1
    done
}

function kill_lightllm_server {
    # Kill LightLLM model HTTP server
    echo '====================='
    echo 'killing model server'
    conda deactivate
    ps aux | grep 'lightllm.server.api_server' | grep -v 'vim' | awk '{print $2}' | xargs kill -9
    wait
}

trap kill_lightllm_server EXIT

install_lightllm

download_weights

start_lightllm_server

ulimit -n 65536 && python ./benchmarks/anyscale/benchmark_llmval.py \
    --host "http://localhost:${PORT}" \
    --model $LOCAL_PATH \
    --num-input-tokens $PROMPT_LEN \
    --num-output-tokens $GEN_LEN \
    --num-users $NUM_USERS \
    --timeout 1200 \
    --max-num-completed-requests $REQUEST_NUM \
    --results-dir "/tmp/cf_range_$(date '+%Y-%m-%d_%H:%M:%S')" \
    --req-type chat