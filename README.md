# llama3-pytorch
fine-tune a Llama 3 using PyTorch FSDP and Q-Lora with the help of Hugging Face TRL, Transformers, peft & datasets. In addition to FSDP we will use Flash Attention v2 through the Pytorch SDPA implementation. 

PyTorch FSDP is a data/model parallelism technique that shards model across GPUs, reducing memory requirements and enabling the training of larger models more efficiently​​​​​​.
Q-LoRA is a fine-tuning method that leverages quantization and Low-Rank Adapters to efficiently reduced computational requirements and memory footprint.

# Install Pytorch for FSDP and FA/SDPA
pip install "torch==2.2.2" tensorboard

# Install Hugging Face libraries
pip install  --upgrade "transformers==4.40.0" "datasets==2.18.0" "accelerate==0.29.3" "evaluate==0.4.1" "bitsandbytes==0.43.1" "huggingface_hub==0.22.2" "trl==0.8.6" "peft==0.10.0"

we need to login into Hugging Face to access the Llama 3 70b model. If you don't have an account yet and accepted the terms, you can create one 

launch the training with the following command:
ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=4 ./scripts/run_fsdp_qlora.py --config llama_3_70b_fsdp_qlora.yaml

