---
title: GKVCache1
emoji: 🐠
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference


tgi_gkv_server extension of text-generaiton-inference from huggingface
changes:
new:
- server/text_generation_server/cache/gkv_cache.py
changes:
- server/text_generation_server/models/model.py
- server/text_generation_server/generator/generator.py
- proto/generate.proto
- clients/python/text_generation.py