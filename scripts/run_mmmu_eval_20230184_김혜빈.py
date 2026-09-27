#!/usr/bin/env python
"""MMMU-val baseline evaluation for Qwen3-VL-4B-Instruct. (3조 - 20230184 김혜빈)

Usage:
    python run_mmmu_eval_20230184_김혜빈.py --output_path results/mmmu_baseline_20230184_김혜빈.jsonl

Model/dataset revisions are pinned to the assignment-fixed values by default
so every run is comparable; override only if you know what you're doing.
"""
import argparse
import ast
import json
import os
import random
import re
import string
from collections import defaultdict

import torch
from datasets import load_dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

SUBJECTS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory",
    "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science",
    "Design", "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics",
    "Energy_and_Power", "Finance", "Geography", "History", "Literature", "Manage",
    "Marketing", "Materials", "Math", "Mechanical_Engineering", "Music", "Pharmacy",
    "Physics", "Psychology", "Public_Health", "Sociology",
]
assert len(SUBJECTS) == 30

OPTION_LETTERS = list(string.ascii_uppercase)

# 생성 설정 — 출처: Qwen/Qwen3-VL-4B-Instruct의 generation_config.json (해당 revision)
GEN_CONFIG = dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0)


def build_prompt_and_images(sample):
    """프롬프트 템플릿 출처: MMMU-Benchmark/MMMU 공식 repo configs/llava1.5.yaml"""
    question_text = sample["question"]
    images = []
    for n in range(1, 8):
        img = sample.get(f"image_{n}")
        if img is not None and f"<image {n}>" in question_text:
            images.append(img)
    if not images:
        for n in range(1, 8):
            img = sample.get(f"image_{n}")
            if img is not None:
                images.append(img)
                break

    clean_q = re.sub(r"<image \d+>", "", question_text).strip()

    if sample["question_type"] == "multiple-choice":
        options = ast.literal_eval(sample["options"])
        option_lines = "\n".join(f"({OPTION_LETTERS[i]}) {opt}" for i, opt in enumerate(options))
        prompt = f"{clean_q}\n\n{option_lines}\n\nAnswer with the option's letter from the given choices directly."
        choices = OPTION_LETTERS[: len(options)]
    else:
        prompt = f"{clean_q}\n\nAnswer the question using a single word or phrase."
        choices = None

    return prompt, images, choices


def parse_multi_choice_response(response, all_choices, index2ans):
    """출처: MMMU-Benchmark/MMMU eval_utils.py의 parse_multi_choice_response 그대로 이식"""
    for ch in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(ch)
    response = " " + response + " "

    index_ans = True
    ans_with_brack = False
    candidates = []
    for choice in all_choices:
        if f"({choice})" in response:
            candidates.append(choice)
            ans_with_brack = True
    if not candidates:
        for choice in all_choices:
            if f" {choice} " in response:
                candidates.append(choice)
    if not candidates and len(response.split()) > 5:
        for idx, ans in index2ans.items():
            if ans.lower() in response.lower():
                candidates.append(idx)
                index_ans = False
    if not candidates:
        return random.choice(all_choices)
    if len(candidates) == 1:
        return candidates[0]

    starts = []
    for c in candidates:
        if index_ans:
            key = f"({c})" if ans_with_brack else f" {c} "
            starts.append(response.rfind(key))
        else:
            starts.append(response.lower().rfind(index2ans[c].lower()))
    return candidates[starts.index(max(starts))]


def check_is_number(s):
    try:
        float(s.replace(",", ""))
        return True
    except ValueError:
        return False


def normalize_str(s):
    s = s.strip()
    if check_is_number(s):
        return [round(float(s.replace(",", "")), 2)]
    s = s.lower()
    return [" " + s, s + " "] if len(s) == 1 else [s]


def extract_numbers(s):
    p1 = r"-?\b\d{1,3}(?:,\d{3})+\b"
    p2 = r"-?\d+(?:\.\d+)?[eE][+-]?\d+"
    p3 = r"-?(?:\d+\.\d+|\.\d+|\d+\b)(?![eE][+-]?\d+)(?![,\d])"
    return re.findall(p1, s) + re.findall(p2, s) + re.findall(p3, s)


def parse_open_response(response):
    """출처: MMMU-Benchmark/MMMU eval_utils.py의 parse_open_response 그대로 이식"""

    def get_key_subresponses(response):
        response = response.strip().strip(".").lower()
        sub_responses = re.split(r"\.\s(?=[A-Z])|\n", response)
        indicators = ["could be ", "so ", "is ", "thus ", "therefore ", "final ", "answer ", "result "]
        key_responses = []
        for index, resp in enumerate(sub_responses):
            local_ind = indicators + ["="] if index == len(sub_responses) - 1 else indicators
            shortest = None
            for ind in local_ind:
                if ind in resp:
                    cand = resp.split(ind)[-1].strip()
                    if not shortest or len(cand) < len(shortest):
                        shortest = cand
            if shortest and shortest not in [":", ",", ".", "!", "?", ";", "'"]:
                key_responses.append(shortest)
        return key_responses if key_responses else [response]

    key_responses = get_key_subresponses(response)
    pred_list = key_responses.copy()
    for resp in key_responses:
        pred_list.extend(extract_numbers(resp))
    tmp = []
    for p in pred_list:
        tmp.extend(normalize_str(p))
    return list(set(tmp))


def eval_open(gold_i, pred_i):
    """출처: MMMU-Benchmark/MMMU eval_utils.py의 eval_open 그대로 이식"""
    norm_answers = []
    for a in gold_i if isinstance(gold_i, list) else [gold_i]:
        norm_answers.extend(normalize_str(a))
    for pred in pred_i:
        if isinstance(pred, str):
            if any(isinstance(na, str) and na in pred for na in norm_answers):
                return True
        elif pred in norm_answers:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL-4B-Instruct MMMU-val baseline evaluation (3조 - 20230184 김혜빈)")
    parser.add_argument("--model_path", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--model_revision", default="ebb281ec70b05090aa6165b016eac8ec08e71b17")
    parser.add_argument("--data_root", default="MMMU/MMMU")
    parser.add_argument("--data_revision", default="98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output_path", required=True, help="결과를 append할 JSONL 경로 (체크포인트 겸용)")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if os.environ.get("HF_TOKEN"):
        from huggingface_hub import login

        login(token=os.environ["HF_TOKEN"])

    torch.manual_seed(args.seed)

    print(f"[1/4] Loading model {args.model_path}@{args.model_revision} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, revision=args.model_revision, dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_path, revision=args.model_revision)
    model.eval()

    print(f"[2/4] Loading {args.data_root} (30 subject configs, split={args.split}) ...")
    datasets_by_subject = {}
    for subj in SUBJECTS:
        ds = load_dataset(args.data_root, subj, split=args.split, revision=args.data_revision)
        assert len(ds) == 30, f"{subj}: expected 30, got {len(ds)}"
        datasets_by_subject[subj] = ds
    total = sum(len(v) for v in datasets_by_subject.values())
    assert total == 900, total
    print(f"    OK: {len(datasets_by_subject)} subjects, {total} questions total")

    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    done_ids = set()
    if os.path.exists(args.output_path):
        with open(args.output_path) as f:
            for line in f:
                done_ids.add(json.loads(line)["id"])
    n_done = len(done_ids)
    if n_done:
        print(f"[3/4] Resuming: {n_done} questions already in {args.output_path}")
    else:
        print("[3/4] Starting fresh run")

    with open(args.output_path, "a") as fout:
        for subj, ds in datasets_by_subject.items():
            for sample in ds:
                qid = sample["id"]
                if qid in done_ids:
                    continue
                prompt, images, choices = build_prompt_and_images(sample)
                content = [{"type": "image", "image": img} for img in images] + [
                    {"type": "text", "text": prompt}
                ]
                messages = [{"role": "user", "content": content}]
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(model.device)
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, **GEN_CONFIG)
                resp = processor.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

                if sample["question_type"] == "multiple-choice":
                    index2ans = {
                        OPTION_LETTERS[i]: opt
                        for i, opt in enumerate(ast.literal_eval(sample["options"]))
                    }
                    pred = parse_multi_choice_response(resp, choices, index2ans)
                    is_correct = pred == sample["answer"]
                else:
                    pred_list = parse_open_response(resp)
                    is_correct = eval_open(sample["answer"], pred_list)
                    pred = str(pred_list)

                record = {
                    "id": qid,
                    "subject": subj,
                    "question_type": sample["question_type"],
                    "response": resp,
                    "pred": pred,
                    "answer": sample["answer"],
                    "correct": is_correct,
                }
                fout.write(json.dumps(record) + "\n")
                fout.flush()

                n_done += 1
                if n_done % 10 == 0:
                    print(f"    {n_done}/900 done ({subj})")

    print("[4/4] Scoring ...")
    correct = defaultdict(int)
    total_c = defaultdict(int)
    with open(args.output_path) as f:
        for line in f:
            r = json.loads(line)
            total_c[r["subject"]] += 1
            if r.get("correct", r["pred"] == r["answer"]):
                correct[r["subject"]] += 1

    per_subject_acc = {s: correct[s] / total_c[s] for s in SUBJECTS if total_c.get(s)}
    macro_avg = sum(per_subject_acc.values()) / len(SUBJECTS)

    print("\n=== Results ===")
    for s in SUBJECTS:
        print(f"{s}: {per_subject_acc.get(s, 0):.3f} ({total_c.get(s, 0)}/30)")
    print(f"\nMacro average: {macro_avg:.4f}  (official: 0.674)")


if __name__ == "__main__":
    main()
