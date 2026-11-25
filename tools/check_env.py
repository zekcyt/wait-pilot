import torch, bitsandbytes, transformers, accelerate

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("device:", torch.cuda.get_device_name(0))
props = torch.cuda.get_device_properties(0)
print(f"VRAM: {props.total_memory / 1024**3:.2f} GiB")
print("capability:", torch.cuda.get_device_capability(0)) 
print("transformers:", transformers.__version__)
print("bitsandbytes:", bitsandbytes.__version__)