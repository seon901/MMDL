"""
09_audit_sheet.py   [새 추론 없음 - GPU 불필요]
'Answer: 줄까지 썼는데 틀린' D유형 오답에서 과목별로 비례하게 N개를 뽑아,
문제 + 이미지 + 모델 출력을 한 화면에서 보며 오답 원인을 분류할 검토 페이지와 태그 CSV를 만든다.
이미 수동 태깅한 30문항(review_tags_draft.csv)은 제외한다.

[실행]
  python3 09_audit_sheet.py --n 40 --zip
[결과 - analysis_out/]
  audit_sample.csv   원인 분류를 적는 표
  audit_review.html  브라우저로 열어서 보는 페이지(이미지는 images/ 폴더에서 읽음)
  audit_images.zip   (--zip 일 때) 표본 이미지만 묶음: 채팅에 올리기 쉽게
"""
import argparse
import csv
import html
import os
import random
import zipfile
from collections import defaultdict

GUIDE = """
<b>root_cause</b> (하나만 고름)<br>
&nbsp;visual_misread : 출력에서 이미지를 서술한 내용이 실제 그림과 다름 (구조·라벨·기호 오독)<br>
&nbsp;knowledge_gap : 이미지 서술은 맞는데 개념·사실 지식이 틀림<br>
&nbsp;calc_formula : 공식 선택 또는 계산·단위 변환이 틀림<br>
&nbsp;reasoning_link : 정보와 지식은 맞는데 비교·연결·결론 단계가 틀림<br>
&nbsp;other : 문항이 모호하거나 정답이 의심스러움, 채점 문제(단위·표기), 이미지 미전달 등<br>
<b>confidence</b> : high / mid / low (전공 지식이 필요해서 확신이 없으면 low)<br>
<b>판단 요령</b> : 모델 출력에서 이미지를 서술한 문장을 먼저 찾아 실제 이미지와 비교한다.
서술이 틀리면 visual_misread, 맞으면 지식·계산·추론 중에서 고른다.
"""


def allocate(counts, total):
    subs = sorted(counts)
    n_all = sum(counts.values())
    raw = {s: counts[s] * total / n_all for s in subs}
    base = {s: min(int(raw[s]), counts[s]) for s in subs}
    for s in subs:  # 항목이 3개 이상인 과목은 최소 1개
        if counts[s] >= 3 and base[s] == 0:
            base[s] = 1
    rem = total - sum(base.values())
    order = sorted(subs, key=lambda s: raw[s] - int(raw[s]), reverse=True)
    while rem > 0:
        moved = False
        for s in order:
            if rem > 0 and base[s] < counts[s]:
                base[s] += 1
                rem -= 1
                moved = True
        if not moved:
            break
    while rem < 0:
        s = max(subs, key=lambda x: base[x])
        base[s] -= 1
        rem += 1
    return base


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--failures", default="failures_to_tag_v2.csv")
    p.add_argument("--exclude", default="review_tags_draft.csv", help="이미 태깅한 문항(제외)")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out_dir", default="analysis_out")
    p.add_argument("--zip", action="store_true", help="표본 이미지만 zip으로 묶기")
    args = p.parse_args()

    with open(args.failures, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    excluded = set()
    for path in (args.exclude, os.path.join(args.out_dir, args.exclude)):
        if os.path.exists(path):
            with open(path, encoding="utf-8-sig") as f:
                excluded |= {r["id"] for r in csv.DictReader(f)}
            break
    pool = [r for r in rows if r["auto_category"].startswith("D.") and r["id"] not in excluded]
    print(f"D유형 {sum(r['auto_category'].startswith('D.') for r in rows)}개 중 "
          f"이미 태깅한 {len(excluded)}개 제외 -> 후보 {len(pool)}개")

    by_subj = defaultdict(list)
    for r in pool:
        by_subj[r["subject"]].append(r)
    plan = allocate({s: len(v) for s, v in by_subj.items()}, min(args.n, len(pool)))
    rnd = random.Random(args.seed)
    picked = []
    for s in sorted(by_subj):
        picked += rnd.sample(by_subj[s], plan[s])
    picked.sort(key=lambda r: (r["subject"], r["id"]))
    print(f"표본 {len(picked)}개 (과목별): " +
          ", ".join(f"{s} {plan[s]}" for s in sorted(plan) if plan[s]))

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "audit_sample.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["no", "subject", "id", "img_type", "gold", "pred", "root_cause", "confidence", "notes"])
        for i, r in enumerate(picked, 1):
            w.writerow([i, r["subject"], r["id"], r["img_type"], r["gold"], r["pred"], "", "", ""])

    parts = ["<html><head><meta charset='utf-8'><title>D유형 감사</title><style>"
             "body{font-family:sans-serif;max-width:1100px;margin:auto;padding:16px}"
             ".card{border:1px solid #ccc;border-radius:8px;padding:12px;margin:18px 0}"
             ".imgs img{max-width:100%;max-height:520px;border:1px solid #ddd;margin:4px}"
             "pre{white-space:pre-wrap;background:#f6f6f6;padding:8px;max-height:380px;overflow:auto}"
             ".ans{font-weight:bold}.guide{background:#fffbe6;padding:10px;border-radius:6px}"
             "</style></head><body>",
             f"<h2>D유형 감사 ({len(picked)}개)</h2><div class='guide'>{GUIDE}</div>"]
    img_files = []
    for i, r in enumerate(picked, 1):
        names = [x for x in r["images"].split("|") if x]
        img_files += names
        imgs = "".join(f"<img src='images/{html.escape(n)}'>" for n in names)
        parts.append(
            f"<div class='card'><h3>#{i} {html.escape(r['id'])} <small>({html.escape(r['img_type'])})</small></h3>"
            f"<div class='imgs'>{imgs}</div>"
            f"<p><b>문제</b><br>{html.escape(r['question'])}</p>"
            f"<p><b>선택지</b> {html.escape(r['options'])}</p>"
            f"<p class='ans'>정답: {html.escape(r['gold'])} &nbsp;|&nbsp; 모델 답: {html.escape(r['pred'])}</p>"
            f"<p><b>모델 출력</b></p><pre>{html.escape(r['raw_output'])}</pre></div>")
    parts.append("</body></html>")
    html_path = os.path.join(args.out_dir, "audit_review.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

    if args.zip:
        zp = os.path.join(args.out_dir, "audit_images.zip")
        missing = 0
        with zipfile.ZipFile(zp, "w") as z:
            for n in img_files:
                src = os.path.join(args.out_dir, "images", n)
                if os.path.exists(src):
                    z.write(src, os.path.join("images", n))
                else:
                    missing += 1
        print(f"zip: {len(img_files) - missing}개 이미지 -> {zp}" + (f" (없는 이미지 {missing}개)" if missing else ""))
    print(f"\n완료: {html_path} (브라우저로 열기), {csv_path} (원인 분류 입력)")


if __name__ == "__main__":
    main()
