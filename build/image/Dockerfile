FROM lazymio/vllm-backport@sha256:349690323ab9aba712111529ed1ca60730199205d8202f67895ffde85b451be3
COPY . /opt/dsv41-offline/
RUN /usr/bin/python3 /opt/dsv41-offline/apply_patches.py
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    VLLM_NO_USAGE_STATS=1 \
    DO_NOT_TRACK=1 \
    VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
    VLLM_USE_V2_MODEL_RUNNER=1 \
    VLLM_DSPARK_FUSED_MARKOV=1
LABEL org.opencontainers.image.title="DeepSeek-V4.1-Flash A100 offline runtime" \
      org.opencontainers.image.revision="a350766628514d597670d2a5217777fdee2ba7f4+offline-patches"
