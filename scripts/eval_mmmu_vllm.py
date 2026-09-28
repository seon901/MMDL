import os
import re
import ast
import time
import json
import argparse

from datasets import load_dataset
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL-4B-Instruct MMMU validation baseline evaluation"
    )
    parser.add_argument(
        "--model_path", default="Qwen/Qwen3-VL-4B-Instruct",
        help="모델 checkpoint 경로 또는 HuggingFace repo id",
    )
    parser.add_argument(
        "--model_revision", default="ebb281ec70b05090aa6165b016eac8ec08e71b17",
        help="모델 revision(commit sha)",
    )
    parser.add_argument(
        "--data_root", default="MMMU/MMMU",
        help="MMMU 데이터 경로. HuggingFace repo id(기본값) 또는 로컬에 다운로드한 데이터셋 경로",
    )
    parser.add_argument(
        "--data_revision", default="98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68",
        help="데이터셋 revision(commit sha). --data_root가 로컬 경로면 무시됨",
    )
    parser.add_argument(
        "--out_dir", default="results_vllm",
        help="결과 저장 폴더",
    )
    return parser.parse_args()


ARGS = parse_args()

SEED = 3407
MAX_NEW_TOKENS = 512
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28

LETTER_POOL = "ABCDEFGHIJKLMNOP"

SUBJECTS = [
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art", "Art_Theory",
    "Basic_Medical_Science", "Biology", "Chemistry", "Clinical_Medicine", "Computer_Science",
    "Design", "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics",
    "Energy_and_Power", "Finance", "Geography", "History", "Literature", "Manage",
    "Marketing", "Materials", "Math", "Mechanical_Engineering", "Music", "Pharmacy",
    "Physics", "Psychology", "Public_Health", "Sociology",
]

OUT_DIR = ARGS.out_dir
os.makedirs(OUT_DIR, exist_ok=True)

def build_prompt_and_images(sample, answer_first=False):
    question_text = sample["question"]
    images_in_order = []
    for i in range(1, 8):
        placeholder = f"<image {i}>"
        img = sample.get(f"image_{i}")
        if placeholder in question_text and img is not None:
            images_in_order.append(img)
    question_text_clean = re.sub(r"<image \d+>", "", question_text).strip()

    options_raw = sample["options"]
    options = ast.literal_eval(options_raw) if isinstance(options_raw, str) else options_raw
    is_mcq = len(options) > 0

    if is_mcq:
        letters = LETTER_POOL[: len(options)]
        opt_lines = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))
        if answer_first:
            prompt_text = (
                f"Question: {question_text_clean}\n"
                f"Options:\n{opt_lines}\n"
                f"First write your final choice on its own line in exactly this format:\n"
                f"Answer: X\n"
                f"where X is a single option letter. Then, after that line, "
                f"briefly explain your reasoning."
            )
        else:
            prompt_text = (
                f"Question: {question_text_clean}\n"
                f"Options:\n{opt_lines}\n"
                f"Answer the question. At the very end of your response, "
                f"write your final choice on a new line in exactly this format:\n"
                f"Answer: X\n"
                f"where X is a single option letter."
            )
    else:
        letters = ""
        prompt_text = (
            f"Question: {question_text_clean}\n"
            f"Answer the question. At the very end of your response, "
            f"write your final answer on a new line in exactly this format:\n"
            f"Answer: <your final answer>\n"
            f"Keep the final answer a short phrase or number, without units or explanation "
            f"unless the question specifically asks for them."
        )
    return prompt_text, images_in_order, letters, is_mcq

def extract_mcq_answer(text, letters):
    if not letters:
        return None
    # "Answer: B", "Answer: **B**", "Answer: (B)" 형태를 모두 허용한다.
    m = re.search(rf"Answer:\s*[*(\[]*\s*([{letters}])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    matches = re.findall(rf"\b([{letters}])\b", text)
    return matches[-1] if matches else None


def extract_open_answer(text):
    """'Answer:' 뒤에 나온 자유 형식 텍스트를 추출. 답이 마지막에 오는 프롬프트이므로 마지막 매칭을 채택한다."""
    matches = re.findall(r"Answer:\s*(.+)", text, re.IGNORECASE)
    if matches:
        return matches[-1].strip().rstrip(".")
    return None

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

def load_subject(subject):
    if os.path.isdir(ARGS.data_root):
        return load_dataset(ARGS.data_root, subject, split="validation")
    return load_dataset(ARGS.data_root, subject, split="validation", revision=ARGS.data_revision)


def build_vllm_input(sample, processor, answer_first=False):
    prompt_text, images, letters, is_mcq = build_prompt_and_images(sample, answer_first)

    content = []
    for img in images:
        content.append({
            "type": "image",
            "image": img,
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
        })
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    image_inputs, _ = process_vision_info(messages)

    vllm_input = {"prompt": text}
    if image_inputs:
        vllm_input["multi_modal_data"] = {"image": image_inputs}

    return vllm_input, letters, is_mcq


def run_subject(subject, llm, processor, sampling_params):
    out_path = os.path.join(OUT_DIR, f"{subject}.json")
    if os.path.exists(out_path):
        print(f"[스킵] {subject} 이미 완료됨 -> {out_path}")
        with open(out_path, encoding="utf-8") as f:
            return json.load(f)

    ds = load_subject(subject)
    t0 = time.time()

    # ----- 1차 실행: 전체 문제 -----
    vllm_inputs, letters_list, is_mcq_list, ids, answers = [], [], [], [], []
    for sample in ds:
        vin, letters, is_mcq = build_vllm_input(sample, processor)
        vllm_inputs.append(vin)
        letters_list.append(letters)
        is_mcq_list.append(is_mcq)
        ids.append(sample["id"])
        answers.append(sample["answer"])

    outputs = llm.generate(vllm_inputs, sampling_params)

    records = []
    for out, letters, is_mcq in zip(outputs, letters_list, is_mcq_list):
        output_text = out.outputs[0].text
        if is_mcq:
            predicted = extract_mcq_answer(output_text, letters)
        else:
            predicted = extract_open_answer(output_text)
        records.append({
            "is_mcq": is_mcq,
            "predicted": predicted,
            "retried": False,
            "raw_output": output_text,
        })

    # ----- 재시도: 답을 추출하지 못한 MCQ만 답-우선 프롬프트로 다시 실행 -----
    retry_idx = [i for i, r in enumerate(records) if r["is_mcq"] and r["predicted"] is None]
    if retry_idx:
        retry_inputs = []
        for i in retry_idx:
            vin, _, _ = build_vllm_input(ds[i], processor, answer_first=True)
            retry_inputs.append(vin)

        retry_outputs = llm.generate(retry_inputs, sampling_params)

        for i, out in zip(retry_idx, retry_outputs):
            output_text = out.outputs[0].text
            records[i].update({
                "predicted": extract_mcq_answer(output_text, letters_list[i]),
                "retried": True,
                "raw_output": output_text,
            })

    # ----- 채점: 재시도까지 끝난 최종 예측에 대해 한 번만 수행 -----
    results = []
    for i, rec in enumerate(records):
        if rec["is_mcq"]:
            correct = rec["predicted"] == answers[i]
        else:
            correct = grade_open(answers[i], rec["predicted"])
        results.append({
            "id": ids[i],
            "subject": subject,
            "is_mcq": rec["is_mcq"],
            "predicted": rec["predicted"],
            "answer": answers[i],
            "correct": correct,
            "retried": rec["retried"],
            "raw_output": rec["raw_output"],
        })

    elapsed = time.time() - t0
    print(f"  {subject}: {len(ds)}문제 처리 완료, MCQ 재시도 {len(retry_idx)}개, {elapsed:.1f}초 "
          f"(평균 {elapsed/len(ds):.2f}초/문제)")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def main():
    print(f"모델: {ARGS.model_path} (revision={ARGS.model_revision})")
    print(f"데이터: {ARGS.data_root}")
    print("모델 로딩 중 (vLLM)...")

    llm = LLM(
        model=ARGS.model_path,
        revision=ARGS.model_revision,
        limit_mm_per_prompt={"image": 8},
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        seed=SEED,
        trust_remote_code=True,
        enforce_eager=True,
        max_num_seqs=4,
    )
    processor = AutoProcessor.from_pretrained(ARGS.model_path, revision=ARGS.model_revision)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.0,
        presence_penalty=1.5,
        max_tokens=MAX_NEW_TOKENS,
        seed=SEED,
    )

    print("모델 로딩 완료.\n")

    all_results = {}
    overall_start = time.time()

    for subject in SUBJECTS:
        print(f"\n=== {subject} 시작 ===")
        results = run_subject(subject, llm, processor, sampling_params)
        all_results[subject] = results

        correct = sum(r["correct"] for r in results)
        acc = correct / len(results)
        open_cnt = sum(1 for r in results if not r["is_mcq"])
        print(f"=== {subject} 완료: {correct}/{len(results)} = {acc:.3f} "
              f"(개방형 {open_cnt}개 포함) ===")

    print("\n\n========== 최종 결과 (vLLM) ==========")
    subject_accs = {}
    for subject, results in all_results.items():
        correct = sum(r["correct"] for r in results)
        subject_accs[subject] = correct / len(results)
        print(f"{subject:40s} {correct:2d}/{len(results):2d}  acc={subject_accs[subject]:.3f}")

    macro_avg = sum(subject_accs.values()) / len(subject_accs)
    total_time = time.time() - overall_start

    retried_total = sum(1 for rs in all_results.values() for r in rs if r.get("retried"))
    recovered_total = sum(1 for rs in all_results.values() for r in rs
                          if r.get("retried") and r["predicted"] is not None)

    print(f"\nOverall (macro average): {macro_avg:.4f}")
    print(f"MCQ 재시도: {retried_total}개 (재시도 후 답 추출 성공 {recovered_total}개)")
    print(f"총 소요 시간: {total_time/60:.1f}분 ({total_time/3600:.2f}시간)")

    summary = {
        "subject_accuracy": subject_accs,
        "overall_macro_avg": macro_avg,
        "total_seconds": round(total_time, 1),
        "backend": "vllm",
        "model_path": ARGS.model_path,
        "data_root": ARGS.data_root,
        "max_new_tokens": MAX_NEW_TOKENS,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "seed": SEED,
        "presence_penalty": 1.5,
        "open_ended_grading": "custom simplified re-implementation of MMMU eval_open",
        "prompt_style": "1st pass: answer-last; retry for MCQ parse failures only: answer-first",
        "retried_mcq": retried_total,
        "retried_recovered": recovered_total,
    }
    with open(os.path.join(OUT_DIR, "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n요약이 {OUT_DIR}/_summary.json 에 저장됨.")


if __name__ == "__main__":
    main()
