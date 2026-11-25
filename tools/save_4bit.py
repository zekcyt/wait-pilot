import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

SRC = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
DST = "F:/models/r1-distill-llama-8b-nf4"

def main():
    tok = AutoTokenizer.from_pretrained(SRC)
    model = AutoModelForCausalLM.from_pretrained(
        SRC, device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16),
    )
    model.save_pretrained(DST)
    tok.save_pretrained(DST)
    print("saved to", DST)

if __name__ == "__main__":
    main()