"""
08_evidence_ab.py   [진단 전용 - 평가 파이프라인 아님]
같은 문항을 (A) 원래 이미지+문제 그대로, (B) 원래 이미지+문제 + 사람이 확인한 '구조/표기 사실'을 붙여서
같은 조건(같은 토큰 예산, 같은 sampling)으로 풀고 비교한다.
  - B에서 회복  -> 그 정보를 뽑아내는 단계(구조/표기 인식)가 병목일 가능성
  - B에서도 실패 -> 이후의 계산/추론/지식이 병목일 가능성
객관식과 개방형 모두 채점한다(채점 규칙은 eval_mmmu_vllm.py와 같은 함수 사용).
A와 B는 항상 같은 max_tokens로 돌린다(기본 2048: 512 잘림 때문에 비교가 가려지는 것을 막기 위함, 진단용).

[입력 CSV] id, group, evidence   (group은 자유 라벨, 예: misread / control)
[실행]
  1) 먼저 모델 없이 점검:  python3 08_evidence_ab.py --evidence evidence_engineering.csv --tag engineering --dry_run
  2) 실제 실행:            python3 08_evidence_ab.py --evidence evidence_engineering.csv --tag engineering --n 5
[결과]
  analysis_out/ab_<tag>.csv           문항별 요약
  analysis_out/ab_<tag>_outputs.jsonl  모든 샘플의 출력 원문(왜 틀렸는지 나중에 볼 수 있음)
문항마다 바로 저장하고, 다시 실행하면 이어서 진행한다.
"""
import argparse
import ast
import csv
import json
import os
import re

from datasets import load_dataset

LETTER_POOL = "ABCDEFGHIJKLMNOP"
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28
SEED = 3407
HEADER = ["id", "group", "type", "n", "A_answered", "A_correct", "A_mean_tokens",
          "B_answered", "B_correct", "B_mean_tokens", "leak_warning"]
PREFIX = ("The following description of the figure was checked by a person who looked at the image:\n")


# ---------- eval_mmmu_vllm.py와 같은 채점 함수 ----------
def normalize_str(s):
    s = str(s).strip()
    variants = {s, s.lower()}
    cleaned = re.sub(r"[,$%\s]", "", s)
    variants.add(cleaned.lower())
    try:
        f = float(cleaned)
        variants.add(str(f))
        if f == int(f):
            variants.add(str(int(f)))
        variants.add(f"{f:.1f}")
        variants.add(f"{f:.2f}")
        variants.add(f"{f:.3f}")
    except ValueError:
        pass
    return variants


def grade_open(gold, pred_text):
    if pred_text is None:
        return False
    gold_val = gold
    if isinstance(gold, str) and gold.strip().startswith("["):
        try:
            gold_val = ast.literal_eval(gold)
        except Exception:
            gold_val = gold
    gold_list = gold_val if isinstance(gold_val, list) else [gold_val]
    pred_norm = normalize_str(pred_text)
    for g in gold_list:
        if normalize_str(g) & pred_norm:
            return True
        if str(g).strip().lower() and str(g).strip().lower() in pred_text.strip().lower():
            return True
    return False


def extract_open_answer(text):
    matches = re.findall(r"Answer:\s*(.+)", text, re.IGNORECASE)
    return matches[-1].strip().rstrip(".") if matches else None


def score(text, gold, letters, is_mcq):
    """(답 줄을 썼는가, 정답인가). 객관식은 'Answer: X' 줄이 있을 때만 답으로 인정(대체 파싱 없음)."""
    if is_mcq:
        m = re.search(rf"Answer:\s*[*(\[]*\s*([{letters}])\b", text, re.IGNORECASE)
        pred = m.group(1).upper() if m else None
        return pred is not None, pred == gold
    pred = extract_open_answer(text)
    return pred is not None, grade_open(gold, pred)


# ---------- 프롬프트 (eval_mmmu_vllm.py의 answer-last 프롬프트와 같은 구성) ----------
def build(sample, evidence=""):
    q = sample["question"]
    imgs = [sample[f"image_{i}"] for i in range(1, 8)
            if f"<image {i}>" in q and sample.get(f"image_{i}") is not None]
    q_clean = re.sub(r"<image \d+>", "", q).strip()
    raw = sample["options"]
    options = ast.literal_eval(raw) if isinstance(raw, str) else raw
    is_mcq = len(options) > 0
    if is_mcq:
        letters = LETTER_POOL[: len(options)]
        opt_lines = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(options))
        body = (f"Question: {q_clean}\nOptions:\n{opt_lines}\n"
                f"Answer the question. At the very end of your response, "
                f"write your final choice on a new line in exactly this format:\n"
                f"Answer: X\nwhere X is a single option letter.")
    else:
        letters = ""
        body = (f"Question: {q_clean}\n"
                f"Answer the question. At the very end of your response, "
                f"write your final answer on a new line in exactly this format:\n"
                f"Answer: <your final answer>\n"
                f"Keep the final answer a short phrase or number, without units or explanation "
                f"unless the question specifically asks for them.")
    if evidence:
        body = PREFIX + evidence.strip() + "\n\n" + body
    return body, imgs, letters, options, is_mcq


# ---------- 증거 누설 점검 (경고일 뿐, 최종 판단은 사람이) ----------
def leak_warnings(evidence, gold, options, question):
    ev = evidence.lower()
    w = []
    gold_val = gold
    if isinstance(gold, str) and gold.strip().startswith("["):
        try:
            gold_val = ast.literal_eval(gold)
        except Exception:
            pass
    for g in (gold_val if isinstance(gold_val, list) else [gold_val]):
        s = str(g).strip().lower()
        if len(options) == 0 and s and re.search(rf"(?<![\w.]){re.escape(s)}(?![\w.])", ev):
            w.append(f"정답값 '{s}'이 증거에 있음")
    if options:
        idx = LETTER_POOL.find(str(gold).strip())
        if 0 <= idx < len(options):
            t = str(options[idx]).strip().lower()
            if len(t) >= 4 and t in ev:
                w.append("정답 선택지 문구가 증거에 있음")
    for k in ("answer", "therefore", "thus", "hence", "equivalent", "total", "so the", "which gives"):
        if re.search(rf"\b{k}\b", ev):
            w.append(f"해석/결론성 단어 '{k}'")
    if "=" in evidence:
        w.append("등호(=)가 있음: 식이나 계산값이 아닌지 확인")
    q_nums = set(re.findall(r"\d+\.?\d*", question + " " + " ".join(map(str, options))))
    extra = sorted(set(re.findall(r"\d+\.?\d*", evidence)) - q_nums)
    if extra:
        w.append("질문에 없는 숫자: " + ", ".join(extra) + " (이미지에서 읽은 값인지 확인)")
    return "; ".join(w)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--evidence", required=True)
    p.add_argument("--tag", default="run")
    p.add_argument("--n", type=int, default=5, help="조건마다 샘플링 횟수")
    p.add_argument("--max_tokens", type=int, default=2048, help="A와 B 모두 동일하게 적용")
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--dry_run", action="store_true", help="모델 없이 문항/누설 경고만 점검")
    p.add_argument("--model_path", default="Qwen/Qwen3-VL-4B-Instruct")
    p.add_argument("--model_revision", default="ebb281ec70b05090aa6165b016eac8ec08e71b17")
    p.add_argument("--data_root", default="MMMU/MMMU")
    p.add_argument("--data_revision", default="98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68")
    p.add_argument("--out_dir", default="analysis_out")
    return p.parse_args()


def load_sample(args, cache, qid):
    subj = re.sub(r"_\d+$", "", re.sub(r"^validation_", "", qid))
    if subj not in cache:
        if os.path.isdir(args.data_root):
            ds = load_dataset(args.data_root, subj, split="validation")
        else:
            ds = load_dataset(args.data_root, subj, split="validation", revision=args.data_revision)
        cache[subj] = {ex["id"]: ex for ex in ds}
    return cache[subj][qid]


def to_input(text, imgs, processor, process_vision_info):
    content = [{"type": "image", "image": im, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}
               for im in imgs]
    content.append({"type": "text", "text": text})
    messages = [{"role": "user", "content": content}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    vin = {"prompt": prompt}
    if image_inputs:
        vin["multi_modal_data"] = {"image": image_inputs}
    return vin


def main():
    args = parse_args()
    with open(args.evidence, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("evidence") or "").strip()]
    print(f"증거가 채워진 문항 {len(rows)}개")
    cache = {}

    if args.dry_run:
        for r in rows:
            ex = load_sample(args, cache, r["id"])
            text, imgs, letters, options, is_mcq = build(ex, r["evidence"])
            warn = leak_warnings(r["evidence"], ex["answer"], options, ex["question"])
            print(f"\n[{r['id'].replace('validation_', '')}] {'객관식' if is_mcq else '개방형'} | "
                  f"그룹 {r.get('group', '')} | 이미지 {len(imgs)}장")
            print("  경고:", warn or "없음")
        print("\n(dry-run 끝: 경고를 확인하고 증거를 고친 뒤 실제 실행)")
        return

    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"ab_{args.tag}.csv")
    jpath = os.path.join(args.out_dir, f"ab_{args.tag}_outputs.jsonl")
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig") as f:
            done = {r["id"] for r in csv.DictReader(f)}
        print(f"이어하기: {len(done)}문항 완료됨")
    new_file = not os.path.exists(path)
    if os.path.exists(jpath):  # 중간에 끊긴 문항의 출력 줄이 남아 있으면 제거(중복 방지)
        keep = []
        with open(jpath, encoding="utf-8") as j0:
            for line in j0:
                try:
                    if json.loads(line)["id"] in done:
                        keep.append(line)
                except Exception:
                    pass
        with open(jpath, "w", encoding="utf-8") as j0:
            j0.writelines(keep)

    processor = AutoProcessor.from_pretrained(args.model_path, revision=args.model_revision)
    llm = LLM(model=args.model_path, revision=args.model_revision,
              limit_mm_per_prompt={"image": 8}, max_model_len=args.max_model_len,
              gpu_memory_utilization=0.90, seed=SEED, trust_remote_code=True,
              enforce_eager=True, max_num_seqs=max(4, args.n))
    sp = SamplingParams(n=args.n, temperature=0.7, top_p=0.8, top_k=20, repetition_penalty=1.0,
                        presence_penalty=1.5, max_tokens=args.max_tokens, seed=SEED)

    with open(path, "a", newline="", encoding="utf-8-sig") as f, \
            open(jpath, "a", encoding="utf-8") as jf:
        w = csv.writer(f)
        if new_file:
            w.writerow(HEADER)
            f.flush()
        for idx, r in enumerate(rows, 1):
            if r["id"] in done:
                continue
            ex = load_sample(args, cache, r["id"])
            gold = ex["answer"]
            stat = {}
            buf = []
            for cond, ev in (("A", ""), ("B", r["evidence"])):
                text, imgs, letters, options, is_mcq = build(ex, ev)
                try:
                    out = llm.generate([to_input(text, imgs, processor, process_vision_info)],
                                       sp, use_tqdm=False)[0]
                except ValueError:
                    stat[cond] = ("SKIP", "SKIP", "")
                    continue
                ans = cor = toks = 0
                for k, o in enumerate(out.outputs):
                    a, c = score(o.text, gold, letters, is_mcq)
                    ans += int(a)
                    cor += int(c)
                    toks += len(o.token_ids)
                    buf.append(json.dumps({"id": r["id"], "cond": cond, "k": k, "answered": a,
                                           "correct": c, "tokens": len(o.token_ids), "text": o.text},
                                          ensure_ascii=False) + "\n")
                stat[cond] = (ans, cor, round(toks / len(out.outputs)))
            jf.writelines(buf)
            jf.flush()
            w.writerow([r["id"], r.get("group", ""), "MCQ" if is_mcq else "open", args.n,
                        stat["A"][0], stat["A"][1], stat["A"][2],
                        stat["B"][0], stat["B"][1], stat["B"][2],
                        leak_warnings(r["evidence"], gold, options, ex["question"])])
            f.flush()
            os.fsync(f.fileno())
            print(f"[{idx}/{len(rows)}] {r['id'].replace('validation_', '')}  "
                  f"정답 A {stat['A'][1]}/{args.n} -> B {stat['B'][1]}/{args.n}  "
                  f"(답 완성 A {stat['A'][0]} -> B {stat['B'][0]})")

    # 요약
    g = {}
    with open(path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if "SKIP" in (r["A_correct"], r["B_correct"]):
                continue
            n = int(r["n"])
            s = g.setdefault(r["group"] or "(그룹 없음)", [0, 0, 0, 0, 0])
            s[0] += 1
            s[1] += int(r["A_correct"]) / n
            s[2] += int(r["B_correct"]) / n
            s[3] += int(r["A_answered"]) / n
            s[4] += int(r["B_answered"]) / n
    print(f"\n=== 결과 (진단용, 평가 조건 아님, max_tokens={args.max_tokens}) ===")
    for k, s in g.items():
        m = s[0]
        print(f"{k}: {m}문항 | 정답률 A {s[1]/m:.0%} -> B {s[2]/m:.0%} | "
              f"답 완성률 A {s[3]/m:.0%} -> B {s[4]/m:.0%}")
    print(f"상세: {path} / 출력 원문: {jpath}")
    print("주의: 문항 수가 적어 방향만 해석. leak_warning이 있는 문항은 증거를 확인.")


if __name__ == "__main__":
    main()
