import os
import openai
import backoff 

completion_tokens = prompt_tokens = 0

api_key = os.getenv("OPENAI_API_KEY", "")
if api_key != "":
    openai.api_key = api_key
else:
    print("Warning: OPENAI_API_KEY is not set")
    
api_base = os.getenv("OPENAI_API_BASE", "")
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base

@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def generate(prompt, model="gpt-4o", temperature=0.7, temp_decay=1, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    if "llama" in model:
        return llama(messages, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
    else:
        return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def chatgpt(messages, model="gpt-4o", temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        # log completion tokens
        completion_tokens += res["usage"]["completion_tokens"]
        prompt_tokens += res["usage"]["prompt_tokens"]
    return outputs

def gpt_usage(backend="gpt-4o"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4o":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}


LOCAL = None

def llama(messages, temperature: float = 0.7, temp_decay: float = 1, max_tokens: int = 1000, n: int = 1, stop: str = None) -> list:
    import torch
    from transformers import pipeline, LogitsProcessorList, LogitsProcessor
    global LOCAL

    if LOCAL is None:
        LOCAL = pipeline(
            "text-generation",
            model="meta-llama/Llama-3.1-8B-Instruct",
            device_map="auto",
        )

    class DecayingTemperatureLogitsWarper(LogitsProcessor):
        def __init__(self, temperature: float, decay: float, reset_token_id: int = 128007):
            self.temperature = temperature
            self.decay = decay
            self.reset_token_id = reset_token_id

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            if self.reset_token_id is not None:
                # get last index of reset token
                reset_token_idx = torch.nonzero(input_ids == self.reset_token_id, as_tuple=False).max()
                # decay temperature
                temp = self.temperature * self.decay ** (input_ids.size(1) - reset_token_idx)
            else:
                temp = self.temperature * self.decay ** input_ids.size(1)

            scores_processed = scores / temp
            return scores_processed
    LOCAL.model._get_logits_processor = lambda *arg, **kwargs : LogitsProcessorList([
        DecayingTemperatureLogitsWarper(temperature, temp_decay, None),
    ])

    outputs = LOCAL(
        messages,
        num_return_sequences=n,
        max_new_tokens=max_tokens,
        stop_strings=stop,
    )
    return [x["generated_text"][-1]["content"] for x in outputs]

