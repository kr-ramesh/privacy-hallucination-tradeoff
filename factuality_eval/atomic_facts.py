import re
import string

import nltk
import numpy as np
import spacy
from nltk.tokenize import sent_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

QUESTION = "Please breakdown the following sentence into independent facts: {}\n"
MONTHS = [m.lower() for m in ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]]
NO_FACT_PREFIXES = ("This sentence does not contain any facts", "Sure", "Please")


class AtomicFactGenerator:
    def __init__(self, lm):
        self.lm = lm
        self.nlp = spacy.load("en_core_web_sm")

    def run(self, generation):
        assert isinstance(generation, str), "generation must be a string"
        paragraphs = [para.strip() for para in generation.split("\n") if len(para.strip()) > 0]
        sentences, para_breaks = [], []
        for idx, paragraph in enumerate(paragraphs):
            if idx > 0:
                para_breaks.append(len(sentences))
            sentences += fix_sentence_splitter(sent_tokenize(paragraph), detect_initials(paragraph))

        atoms = {}
        pairs = []
        for i, sent in enumerate(sentences):
            if sent.startswith(NO_FACT_PREFIXES) or (i == 0 and sent.startswith("Here are")):
                pairs.append((sent, []))
                continue
            if sent not in atoms:
                atoms[sent] = text_to_sentences(self.lm.generate(QUESTION.format(sent)))
            pairs.append((sent, atoms[sent]))
        return postprocess_atomic_facts(pairs, para_breaks, self.nlp)


def text_to_sentences(text):
    text = re.sub(r"\*\*(.*?)\*\*", "", text)
    sentences = [part.strip() for part in re.split(r"(?:^|\n)(?:- |\d+\.\s*)", text)[1:] if part.strip()]
    if sentences and sentences[-1][-1] not in {".", "!", "?"}:
        sentences[-1] += "."
    return sentences


def normalize_answer(s):
    s = "".join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def is_num(text):
    try:
        int(text)
        return True
    except Exception:
        return False


def is_date(text):
    return all(is_num(token) or token in MONTHS for token in normalize_answer(text).split(" "))


def detect_entities(text, nlp):
    entities = set()

    def add(ent_text):
        entities.update(t.strip() for t in ent_text.split("-")) if "-" in ent_text else entities.add(ent_text)

    for ent in nlp(text).ents:
        if ent.label_ in ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]:
            if is_date(ent.text):
                add(ent.text)
            else:
                for token in ent.text.split():
                    if is_date(token):
                        add(token)
    for new_ent in set(re.findall(r"\b\d+\b", text)):
        if not np.any([new_ent in ent for ent in entities]):
            entities.add(new_ent)
    return entities


def postprocess_atomic_facts(_atomic_facts, para_breaks, nlp):
    verbs = ["born.", " appointed.", " characterized.", " described.", " known.", " member.", " advocate.", "served.", "elected."]
    permitted_verbs = ["founding member."]

    atomic_facts, new_para_breaks = [], []
    for i, (sent, facts) in enumerate(_atomic_facts):
        sent = sent.strip()
        if len(sent.split()) == 1 and i not in para_breaks and i > 0:
            atomic_facts[-1][0] += " " + sent
            atomic_facts[-1][1] += facts
        else:
            if i in para_breaks:
                new_para_breaks.append(len(atomic_facts))
            atomic_facts.append([sent, facts])

    new_atomic_facts = []
    for sent, facts in atomic_facts:
        entities = detect_entities(sent, nlp)
        covered_entities, new_facts = set(), []
        for i, fact in enumerate(facts):
            if any(fact.endswith(v) for v in verbs) and not any(fact.endswith(v) for v in permitted_verbs):
                if any(fact[:-1] in other for j, other in enumerate(facts) if j != i):
                    continue
            sent_entities = detect_entities(fact, nlp)
            covered_entities |= {e for e in sent_entities if e in entities}
            skip = False
            for new_ent in sent_entities - entities:
                pre_ent = next((ent for ent in entities if ent.startswith(new_ent)), None)
                if pre_ent is None:
                    skip = True
                    break
                fact = fact.replace(new_ent, pre_ent)
                covered_entities.add(pre_ent)
            if skip or fact in new_facts:
                continue
            new_facts.append(fact)
        new_atomic_facts.append((sent, new_facts if entities == covered_entities else facts))
    return new_atomic_facts, new_para_breaks


def detect_initials(text):
    return re.findall(r"[A-Z]\. ?[A-Z]\.", text)


def fix_sentence_splitter(curr_sentences, initials):
    for initial in initials:
        if not np.any([initial in sent for sent in curr_sentences]):
            alpha1, alpha2 = [t.strip() for t in initial.split(".") if len(t.strip()) > 0]
            for i, (sent1, sent2) in enumerate(zip(curr_sentences, curr_sentences[1:])):
                if sent1.endswith(alpha1 + ".") and sent2.startswith(alpha2 + "."):
                    curr_sentences = curr_sentences[:i] + [curr_sentences[i] + " " + curr_sentences[i + 1]] + curr_sentences[i + 2:]
                    break
    sentences, combine_with_previous = [], None
    for sent_idx, sent in enumerate(curr_sentences):
        if len(sent.split()) <= 1 and sent_idx == 0:
            combine_with_previous = True
            sentences.append(sent)
        elif len(sent.split()) <= 1:
            sentences[-1] += " " + sent
        elif sent[0].isalpha() and not sent[0].isupper() and sent_idx > 0:
            sentences[-1] += " " + sent
            combine_with_previous = False
        elif combine_with_previous:
            sentences[-1] += " " + sent
            combine_with_previous = False
        else:
            sentences.append(sent)
    return sentences
