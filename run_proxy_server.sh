#!/bin/bash
python load_balance_proxy_server_example.py \
    --host 61.28.30.29 \
    --port 8080 \
    --prefiller-hosts 61.28.30.29 \
    --prefiller-port 13700 \
    --decoder-hosts 61.28.30.29 \
    --decoder-ports 13701