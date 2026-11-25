import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    print(f"VRAM after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    messages = [{"role": "user", "content":
                 "What is 17 * 23? Please reason step by step, "
                 "and put your final answer within \\boxed{}."}]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    out = model.generate(**inputs, max_new_tokens=2048,
                         do_sample=True, temperature=0.6, top_p=0.95)
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    print(tok.decode(new_tokens, skip_special_tokens=False))

if __name__ == "__main__":
    main()