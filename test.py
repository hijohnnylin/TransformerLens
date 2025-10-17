import torch

from transformer_lens.model_bridge.sources.transformers import boot

model = boot(
    # CASE 1: gemma-3-270m-it errors with AttributeError: 'Gemma3MLP' object has no attribute 'ln'
    #   Stack -> architecture_adapter.py", line 114, in get_remote_component -> current = getattr(current, part)
    model_name="google/gemma-3-270m-it",
    # CASE 2: gemma-2-2b-it works, but gives me gibberish (even capping torch at 2.7.1)
    # model_name="google/gemma-2-2b-it",
    # CASE 3: Llama-3.1-8B-Instruct works fine
    # model_name="meta-llama/Llama-3.1-8B-Instruct",
    device="mps",  # have also tried "cpu"
    dtype=torch.bfloat16,
)

model.enable_compatibility_mode()  # no_processing=True)
messages = [{"role": "user", "content": "hi"}]
chat_text = model.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, prepend_bos=False
)
tokens = model.to_tokens(chat_text, prepend_bos=False)
print(tokens)
for result in model.generate_stream(
    tokens, max_tokens_per_yield=1, do_sample=False, use_past_kv_cache=False
):
    # result is a tensor of token IDs for the batch
    # decode expects a list of integers, so convert tensor to list
    if isinstance(result, torch.Tensor):
        # Convert tensor to list of integers
        token_ids = result.squeeze().tolist()
        # If it's a single token, tolist() returns an int, wrap it in a list
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        print(model.tokenizer.decode(token_ids))
    elif isinstance(result, list):
        # If already a list, decode it directly
        print(model.tokenizer.decode(result))
    else:
        # Single token ID
        print(model.tokenizer.decode([result]))
