"""Zero-shot LLM career classification (no training data), scored by label likelihood. Needs a GPU.

For each text the model sees the class list with one-line definitions, then every class name is
scored by its log-probability as the answer; the best-scoring class is the prediction. This never
produces an invalid label and also gives a ranking for top-3.

Default model: Qwen/Qwen2.5-3B-Instruct (Qwen Research License, research use; fits a 16 GB T4 in fp16).
Evaluated on a stratified sample (per class) of each chosen test fold (default fold 0) and on the Pakistani forms.
"""
import argparse, re
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import common as C

FARUQUE_DEF = {
    "Artificial Intelligence": "builds AI and machine learning models",
    "Data Science": "analyses data, statistics and visualisation",
    "Development": "builds web or mobile applications",
    "Security": "protects systems and networks (cybersecurity)",
    "Software Development and Engineering": "designs and engineers software systems in general",
    "User Experience (UX) and User Interface (UI) Design": "designs user interfaces and user experience",
}


def short_def(text, words=22):
    first = re.split(r"(?<=[.])\s", str(text))[0]
    w = first.split()
    return " ".join(w[:words]) + ("..." if len(w) > words else "")


def definitions(classes):
    ls = C.label_set().set_index("class")
    out = {}
    for c in classes:
        if c in ls.index:
            out[c] = short_def(ls.loc[c, "esco_definition"] or ls.loc[c, "onet_definition"])
        else:
            out[c] = FARUQUE_DEF.get(c, "")
    return out


PROMPT_VARIANT = "default"   # set from --prompt_variant


def build_prompt(tok, text, classes, defs, max_words):
    """default: field names with one-line ESCO/O*NET definitions (as in the paper)
    nodef:   field names only, same wording
    plain:   definitions kept, different instruction wording (a job-matching framing)"""
    words = str(text).split()
    text = " ".join(words[:max_words])
    if PROMPT_VARIANT == "nodef":
        options = "\n".join(f"- {c}" for c in classes)
    else:
        options = "\n".join(f"- {c}: {defs[c]}" for c in classes)
    if PROMPT_VARIANT == "plain":
        msgs = [
            {"role": "system", "content": "You match people to job fields in computing."},
            {"role": "user", "content": f"Here is a text written by a person:\n\"\"\"{text}\"\"\"\n\nPossible job fields:\n{options}\n\nReply with the one job field from the list that this person most likely works in or wants to work in."},
        ]
    else:
        msgs = [
            {"role": "system", "content": "You are a career counsellor for computing students. Classify the person into exactly one career field."},
            {"role": "user", "content": f"Career fields:\n{options}\n\nPerson's description:\n\"\"\"{text}\"\"\"\n\nWhich career field fits this person best? Answer with the field name only."},
        ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def score(model, tok, prompt, classes, end_text, dev):
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    cand = [tok(c + end_text, add_special_tokens=False)["input_ids"] for c in classes]
    seqs = [p_ids + c for c in cand]
    L = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    in_dev = model.get_input_embeddings().weight.device          # works for one GPU and for device_map="auto"
    ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device=in_dev)
    att = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device=in_dev)
    # run the transformer once, then apply the output layer only where answer tokens are predicted
    # (full-vocabulary logits for every position would need several GB on a 3B model)
    hidden = model.base_model(input_ids=ids, attention_mask=att).last_hidden_state
    head = model.get_output_embeddings(); hdev = head.weight.device
    hidden = hidden.to(hdev)
    out = []
    for i, c in enumerate(cand):
        pos = torch.arange(len(p_ids) - 1, len(p_ids) - 1 + len(c), device=hdev)   # position t predicts token t+1
        logp = torch.log_softmax(head(hidden[i, pos]).float(), -1)
        out.append(logp[torch.arange(len(c), device=hdev), torch.tensor(c, device=hdev)].sum().item())
    return np.array(out)


def evaluate(model, tok, dev, df, classes, name, a):
    defs = definitions(classes); end_text = a.end_text
    S = []
    for i, t in enumerate(df.text):
        S.append(score(model, tok, build_prompt(tok, t, classes, defs, a.max_words), classes, end_text, dev))
        if (i + 1) % 50 == 0: print(f"  {name}: {i + 1}/{len(df)}")
    S = np.vstack(S); P = np.exp(S - S.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
    pred = np.array(classes)[S.argmax(1)]
    m = C.metrics(df.label, pred, P, classes)
    p = pd.DataFrame({"id": df.id, "true": df.label, "pred": pred})
    C.save(a.tag, name, m, preds=p, extra={"top_confusions": C.top_confusions(p.true, p.pred), "config": vars(a)})


def sample_fold(df, per_class, fold=0):
    """stratified sample of one test fold (fold 0 gives exactly the sample used in the paper)"""
    t = df[df.fold == fold]
    return pd.concat([g.sample(min(len(g), per_class), random_state=0) for _, g in t.groupby("label")]).reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--tag", default="llm_qwen2.5-3b_zeroshot")
    ap.add_argument("--datasets", default="djinni,faruque,forms", help="any of djinni,stackoverflow,faruque,forms")
    ap.add_argument("--per_class", type=int, default=40)
    ap.add_argument("--max_words", type=int, default=250)
    ap.add_argument("--end_text", default="<|im_end|>", help="end-of-answer marker of the chat template")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt_variant", default="default", choices=["default", "nodef", "plain"])
    ap.add_argument("--folds", default="0", help="test folds to sample, e.g. 0 or 1,2,3,4; one result file per fold")
    ap.add_argument("--device_map", default="", help='"auto" spreads a large model over all GPUs (e.g. 7B on 2x T4)')
    a = ap.parse_args()
    PROMPT_VARIANT = a.prompt_variant
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    dtype = torch.float16 if dev == "cuda" else torch.float32
    if a.device_map:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype, device_map=a.device_map).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).to(dev).eval()
    for ds in a.datasets.split(","):
        if ds == "forms" and not C.has_forms():
            print("forms_test.csv not found (pilot data is not public); skipping"); continue
        if ds == "forms":
            classes = sorted(C.label_set()["class"])           # zero-shot needs no training data: all 13 classes
            df = C.load_forms().reset_index(drop=True)
            name = "forms_zeroshot"
            if a.limit: df = df.head(a.limit)
            evaluate(model, tok, dev, df, classes, name, a); continue
        full = C.load(ds); classes = sorted(full.label.unique())
        for k in [int(x) for x in a.folds.split(",")]:
            df = sample_fold(full, a.per_class, k); name = f"{ds}_fold{k}_sample"
            if a.limit: df = df.head(a.limit)
            evaluate(model, tok, dev, df, classes, name, a)
