from vllm import LLM, SamplingParams


def generate():
    prompts = [
        "Generate a positive movie review.",
        "Generate a negative movie review.",
        "Generate a news article about the economy.",
    ]
    sampling_params = SamplingParams(temperature=0.9, max_tokens=128)
    llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.1")
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

if __name__ == "__main__":
    generate()