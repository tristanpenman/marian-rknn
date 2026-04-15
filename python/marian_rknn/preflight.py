#!/usr/bin/env python3

import argparse

from transformers import MarianMTModel, MarianTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="Which device to run the model on (e.g. cpu or cuda)")
    parser.add_argument("--model-name", default="Helsinki-NLP/opus-mt-en-fr", help="Name of the model to use")
    return parser.parse_args()

def main():
    params = parse_args()
    print(f"Using device: {params.device}")
    print(f"Using model: {params.model_name}")

    tokenizer = MarianTokenizer.from_pretrained(params.model_name)
    model = MarianMTModel.from_pretrained(params.model_name).to(params.device)
    print("Enter text to translate (empty line to quit):")
    while True:
        line = input("> ").strip()
        if not line:
            break
        input_ids = tokenizer([line], return_tensors="pt", padding=True).to(params.device)
        tokens = model.generate(**input_ids)
        output = [tokenizer.decode(t, skip_special_tokens=True) for t in tokens]
        print(output[0])

if __name__ == "__main__":
    main()
