import os
import string

import numpy as np

from .abstain_detection import is_response_abstained
from .atomic_facts import AtomicFactGenerator
from .lm import DeepSeekLM, MetaLlamaLM, MistralLM
from .retrieval import DocDB, Retrieval

MODELS = [
    ("retrieval+mistral", MistralLM, "mistralai/Mistral-7B-Instruct-v0.2"),
    ("retrieval+biomistral", MistralLM, "BioMistral/BioMistral-7B-DARE"),
    ("retrieval+metallama", MetaLlamaLM, "meta-llama/Llama-3.1-8B-Instruct"),
    ("retrieval+metallama-mini", MetaLlamaLM, "meta-llama/Llama-3.2-3B-Instruct"),
    ("retrieval+deepseek", DeepSeekLM, "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"),
]
MISTRAL_TRUE_ID, MISTRAL_FALSE_ID = 6110, 8250


def _core_result_parser(_rs):
    from core.utils.instances import CoreInstance
    return [CoreInstance(text=fact, sent=sent, topic=None) for sent, facts in _rs[0] for fact in facts]


def _core_result_merger(_selected, _inputs, _rs):
    result = _rs[0]
    outputs = {sent: [] for sent, _ in result}
    for idx in _selected:
        assert _inputs[idx].sent in outputs, f"{_inputs[idx].sent} not in {outputs.keys()}"
        outputs[_inputs[idx].sent].append(_inputs[idx].text)
    return [(sent, outputs[sent]) for sent, _ in result], result[0]


class FactScorer:
    def __init__(self, model_name="retrieval+metallama", cache_dir=".cache/factscore", db_path=None, abstain_detection_type=None, use_core=True):
        names = [name for name, _, _ in MODELS]
        assert model_name in names, f"model_name must be one of {names}"
        print(f"Using {model_name}")
        self.model_name, self.cache_dir, self.db_path, self.abstain_detection_type = model_name, cache_dir, db_path, abstain_detection_type
        os.makedirs(cache_dir, exist_ok=True)

        lm_class, model_dir = next((cls, path) for name, cls, path in MODELS if name in model_name)
        self.lm = lm_class(model_dir)
        self.lm.load_model()
        self.af_generator = AtomicFactGenerator(self.lm)
        if use_core:
            from core import Core
            self.af_generator.run = Core(result_parser=_core_result_parser, result_merger=_core_result_merger)(self.af_generator.run)
        self.db, self.retrieval = {}, {}

    def register_knowledge_source(self, name, db_path=None):
        assert name not in self.retrieval, f"{name} already registered"
        self.db[name] = DocDB(db_path or self.db_path)
        self.retrieval[name] = Retrieval(self.db[name], os.path.join(self.cache_dir, f"retrieval-{name}.json"), os.path.join(self.cache_dir, f"retrieval-{name}.pkl"))

    def save_cache(self):
        for retrieval in self.retrieval.values():
            retrieval.save_cache()

    def get_atomic_facts(self, generations):
        atomic_facts = []
        for generation in generations:
            if is_response_abstained(generation, self.abstain_detection_type):
                atomic_facts.append(None)
                continue
            pairs, _ = self.af_generator.run(generation)
            facts = [fact for _, sent_facts in pairs for fact in sent_facts]
            atomic_facts.append(facts if facts else None)
        return atomic_facts

    def score_facts(self, topics, generations, atomic_facts, gamma=10, knowledge_source="enwiki-20230401"):
        if knowledge_source not in self.retrieval:
            self.register_knowledge_source(knowledge_source)
        assert len(atomic_facts) == len(topics) == len(generations)

        scores, init_scores, decisions = [], [], []
        for topic, facts in zip(topics, atomic_facts):
            if facts is None:
                decisions.append(None)
                continue
            try:
                decision = self._get_score(topic, facts, knowledge_source)
            except Exception:
                print("Topic: ", topic, " does not exist in the database. Continuing anyway.")
                scores.append(0)
                init_scores.append(0)
                decisions.append([])
                continue
            score = np.mean([d["is_supported"] for d in decision])
            if gamma:
                init_scores.append(score)
                score = score * (1.0 if len(facts) > gamma else np.exp(1 - gamma / len(facts)))
            decisions.append(decision)
            scores.append(score)
            if len(scores) % 10 == 0:
                self.save_cache()
        self.save_cache()

        out = {"score": np.mean(scores),
               "scores": scores,
               "respond_ratio": np.mean([facts is not None for facts in atomic_facts]),
               "decisions": decisions,
               "num_facts_per_response": [len(d) for d in decisions if d is not None],
               "num_facts_per_response_aggregated": np.mean([len(d) for d in decisions if d is not None])}
        if gamma:
            out["init_scores"], out["init_score"] = init_scores, np.mean(init_scores)
        return out

    def get_score(self, topics, generations, gamma=10, knowledge_source=None):
        if isinstance(topics, str) and isinstance(generations, str):
            topics, generations = [topics], [generations]
        assert isinstance(topics, list) and isinstance(generations, list), "`topics` and `generations` should be lists."
        assert len(topics) == len(generations), "`topics` and `generations` should have the same length"
        knowledge_source = knowledge_source or "enwiki-20230401"
        if knowledge_source not in self.retrieval:
            self.register_knowledge_source(knowledge_source)
        return self.score_facts(topics, generations, self.get_atomic_facts(generations), gamma, knowledge_source)

    def _get_score(self, topic, atomic_facts, knowledge_source):
        decisions = []
        for atom in atomic_facts:
            _, indices = self.retrieval[knowledge_source].get_passages(topic, atom, k=5)
            passages = self.db[knowledge_source].get_text_from_title(topic)
            context = "".join("Title: {}\nText: {}\n\n".format(topic, passages[i]["text"].replace("<s>", "").replace("</s>", "")) for i in reversed(indices))
            definition = "Answer the question about {} based on the given context.\n\n".format(topic) + context.strip()
            if definition[-1] not in string.punctuation:
                definition += "."
            prompt = "{}\n\nInput: {} True or False?\nOutput:".format(definition.strip(), atom.strip())
            decisions.append({"atom": atom, "is_supported": self._is_supported(prompt)})
        return decisions

    def _is_supported(self, prompt):
        if "retrieval+mistral" in self.model_name:
            output = self.lm._generate(prompt, max_new_tokens=1)
        elif "retrieval+deepseek" in self.model_name:
            output = self.lm.generate(prompt, max_new_tokens=1000)
        else:
            output = self.lm.generate(prompt)
        if isinstance(output, np.ndarray) and "retrieval+mistral" in self.model_name:
            return bool(output[0, MISTRAL_TRUE_ID] > output[0, MISTRAL_FALSE_ID])
        answer = output.lower()
        if "true" in answer or "false" in answer:
            if "true" in answer and "false" not in answer:
                return True
            if "false" in answer and "true" not in answer:
                return False
            return answer.index("true") > answer.index("false")
        return all(keyword not in answer.translate(str.maketrans("", "", string.punctuation)).split() for keyword in ["not", "cannot", "unknown", "information"])
