import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .few_shot_examples import DEMOS


class HFChatLM:
    response_role = "assistant"
    add_generation_prompt = False

    def __init__(self, model_dir):
        self.model_dir = model_dir

    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, device_map="auto", torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def postprocess(self, text):
        return text

    def generate(self, q, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.95):
        messages = [m for demo_q, demo_r in DEMOS for m in ({"role": "user", "content": demo_q}, {"role": self.response_role, "content": demo_r})]
        messages.append({"role": "user", "content": q})
        tokens = self.tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=self.add_generation_prompt).to("cuda")
        generated = self.model.generate(tokens, pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=max_new_tokens,
                                        do_sample=do_sample, temperature=temperature, top_p=top_p)
        return self.postprocess(self.tokenizer.batch_decode(generated)[0])

    def _generate(self, prompt, max_new_tokens=1):
        tokens = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], return_tensors="pt", add_generation_prompt=True).to("cuda")
        out = self.model.generate(tokens, pad_token_id=self.tokenizer.eos_token_id, max_new_tokens=max_new_tokens,
                                  output_scores=True, return_dict_in_generate=True)
        return out["scores"][0].detach().cpu().numpy()


class MistralLM(HFChatLM):
    def postprocess(self, text):
        return re.split(r"\[/INST\]", text)[-1].replace("</s>", "")


class MetaLlamaLM(HFChatLM):
    response_role = "system"

    def postprocess(self, text):
        return re.split(r"<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>", text)[-1].replace("<|eot_id|>", "").strip()


class DeepSeekLM(HFChatLM):
    add_generation_prompt = True

    def postprocess(self, text):
        return re.split(r"</think>", text)[-1].replace("<｜end▁of▁sentence｜>", "").strip()
