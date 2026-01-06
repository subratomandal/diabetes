# v 2.0
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import sys
import traceback

WORKING_MODEL_NAME = "google/flan-t5-large"

def load_model(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        dtype = torch.float16 if use_cuda else torch.float32
        device_map_config = "auto" if use_cuda else None

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map_config
        )

        return model, tokenizer, device
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        raise

def generate_response(model, tokenizer, device, prompt, max_len=256):
    input_prompt = (
        "You are a knowledgeable medical assistant specializing in diabetes. "
        f"Please provide a detailed, coherent, and accurate response to the question below:\n{prompt}"
    )

    try:
        inputs = tokenizer(
            input_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            output = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_len,
                num_beams=6,
                no_repeat_ngram_size=4,
                length_penalty=1.3,
                early_stopping=True,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return f"Error during text generation: {str(e)}"

def main():
    try:
        model, tokenizer, device = load_model(WORKING_MODEL_NAME)
    except Exception as e:
        sys.exit(1)

    print("Model loaded and ready for input.", flush=True)
    while True:
        try:
            query = input()
            query = query.strip()
            if not query: continue

            response = generate_response(model, tokenizer, device, query)
            
            print(response, flush=True)
            print("END_OF_RESPONSE", flush=True)

        except EOFError:
            break
        except KeyboardInterrupt:
            break
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            print(f"Error occurred: {e}", flush=True)
            print("END_OF_RESPONSE", flush=True)

if __name__ == "__main__":
    main()
