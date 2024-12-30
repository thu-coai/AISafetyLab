deepspeed --include localhost:0,1,2,3 --master_port=20959 main.py \
    --config examples/safe_unlearning/default_config.yaml