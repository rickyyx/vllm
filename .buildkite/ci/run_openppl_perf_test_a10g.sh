#!/usr/bin/env bash

# The below benchmarks run for ppl.llm.serving inference engine

NAME="default"
MODEL="7b"
TP="1"
PROMPT_LEN="512"
GEN_LEN="128"
REQUEST_NUM="10"
NUM_USERS="1"
DOWNLOAD_DIR="/mnt/local_storage"
MAX_NUM_BATCHED_TOKENS="5120"
PORT="8002"

INSTALL_S3_PATH="s3://vllm-ci-opensource-perf"
MODEL_S3_PATH="${INSTALL_S3_PATH}/ppl.llm.serving/llama-${MODEL}-pmx/"
LOCAL_PATH="${DOWNLOAD_DIR}/llama-${MODEL}-pmx/"

function install_openppl {
    # Build ppl.llm.serving
    aws s3 cp "${INSTALL_S3_PATH}/llmval.tar.gz" $DOWNLOAD_DIR
    pip install "${DOWNLOAD_DIR}/llmval.tar.gz"
    pushd ..
    git clone https://github.com/openppl-public/ppl.llm.serving.git
    pushd ppl.llm.serving
    git reset 0c654bf362437303ef846a1858644120eb9e1f58 --hard
    ./build.sh -DPPLNN_USE_LLM_CUDA=ON -DPPLNN_CUDA_ENABLE_NCCL=ON -DPPLNN_ENABLE_CUDA_JIT=OFF -DPPLNN_CUDA_ARCHITECTURES="'80;86;87'" -DPPLCOMMON_CUDA_ARCHITECTURES="'80;86;87'"
    popd
    popd
    sudo apt-get install netcat -y
}

function download_weights {
    # Download pre-compiled ppl.pmx llama model from S3
    if [ ! -f "${LOCAL_PATH}/download_weights.finished" ]
    then
        echo '========================='
        echo "downloading model weights from ${MODEL_S3_PATH}"
        mkdir -p $LOCAL_PATH
        echo "Downloading weights to ${LOCAL_PATH}"
        aws s3 cp $MODEL_S3_PATH $LOCAL_PATH --recursive
        echo "Downloading llmval legacy benchmark script with openppl grpc support" 
        aws s3 cp "${INSTALL_S3_PATH}/benchmark_llmval.py" benchmarks/anyscale/
        touch "${LOCAL_PATH}/download_weights.finished"
        echo '========================='
        echo 'model weights downloaded'
    else
        echo '================================'
        echo "model weights already downloaded in ${LOCAL_PATH}"
    fi
}

function start_openppl_server {
    # Start ppl.llm.serving model gRPC server
    echo '====================='
    echo 'starting model server'
    ulimit -n 65536 && ../ppl.llm.serving/ppl-build/ppl_llm_server "${LOCAL_PATH}/llama_${MODEL}.json" \
        2>&1 &

    echo "Waiting openppl server to launch on ${PORT}..."

    while ! nc -z localhost ${PORT}; do   
        sleep 0.1
    done
}

function kill_openppl_server {
    # Kill ppl.llm.serving model gRPC server
    echo '====================='
    echo 'killing model server'
    ps aux | grep 'ppl_llm_server' | grep -v 'vim' | awk '{print $2}' | xargs kill -9
    wait
}

trap kill_openppl_server EXIT

install_openppl

download_weights

start_openppl_server

ulimit -n 65536 && python ./benchmarks/anyscale/benchmark_llmval.py \
    --host "http://localhost:${PORT}" \
    --model $LOCAL_PATH \
    --num-input-tokens $PROMPT_LEN \
    --num-output-tokens $GEN_LEN \
    --num-users $NUM_USERS \
    --timeout 1200 \
    --max-num-completed-requests $REQUEST_NUM \
    --results-dir "/tmp/cf_range_$(date '+%Y-%m-%d_%H:%M:%S')" \
    --req-type openppl
