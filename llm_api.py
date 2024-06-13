import torch
import openai

class llama_generate_api:
  def __init__(self, llm, tokenizer):
    self.llm = llm
    self.tokenizer = tokenizer

  def __call__(self, prompts, ended, past_key_values=None, collect_mode=False):
    return_text = [""] * len(prompts)

    input_prompts = []
    input_prompts_ids = []
    for i, prompt in enumerate(prompts):
        if not ended[i]:
            input_prompts.append(prompt)
            input_prompts_ids.append(i)

    input_ids = self.tokenizer(input_prompts,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            ).input_ids.cuda()

    generated_ids = self.llm.generate(input_ids=input_ids, max_new_tokens=20 if collect_mode else 100)
    generated_ids = generated_ids[:, input_ids.shape[-1]:]
    gen_txt = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    
    for j, txt in enumerate(gen_txt):
        return_text[input_prompts_ids[j]] = txt.split('\n')[0]
        
    return return_text, past_key_values