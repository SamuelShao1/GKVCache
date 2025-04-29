FROM ghcr.io/huggingface/text-generation-inference:3.2.3

RUN mkdir -p /opt/conda/lib/python3.11/site-packages/text_generation_server/cache

COPY ./text-generation-inference/server/text_generation_server/cache/kv_cache.py /opt/conda/lib/python3.11/site-packages/text_generation_server/cache/
COPY ./text-generation-inference/server/text_generation_server/cache/__init__.py /opt/conda/lib/python3.11/site-packages/text_generation_server/cache/

COPY ./text-generation-inference/server/text_generation_server/models/model.py /opt/conda/lib/python3.11/site-packages/text_generation_server/models/
COPY ./text-generation-inference/server/text_generation_server/models/causal_lm.py /opt/conda/lib/python3.11/site-packages/text_generation_server/models/

COPY ./text-generation-inference/server/text_generation_server/server.py /opt/conda/lib/python3.11/site-packages/text_generation_server/