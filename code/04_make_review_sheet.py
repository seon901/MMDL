"""
04_make_review_sheet.py
'답은 했는데 틀린' 오답을 과목별로 N개씩 뽑아,
(1) 이미지·문제·모델 출력을 한 화면에 보여주는 HTML 검토 페이지와
(2) 태그를 적을 CSV를 만든다. GPU나 모델은 쓰지 않는다.

[실행]
  python3 04_make_review_sheet.py --failures failures_to_tag_v2.csv \
      --subjects Biology Diagnostics_and_Laboratory_Medicine Agriculture --per_subject 10
[결과]
  analysis_out/review.html  : 브라우저로 열어서 보는 용도
  analysis_out/review_tags.csv : 엑셀로 열어 failure_type 등을 채우는 용도
"""
import argparse
import csv
import html
import os

GUIDE = """
<b>failure_type</b> (하나만 고름)<br>
&nbsp;perception : 이미지의 구조·형태·특징을 잘못 봄 (예: 병리 조직 형태, 잎의 병변을 다른 것으로 서술)<br>
&nbsp;ocr : 이미지 속 글자·숫자·라벨을 잘못 읽음<br>
&nbsp;knowledge : 이미지는 맞게 서술했는데 개념·전공지식이 틀림<br>
&nbsp;reasoning : 정보와 개념은 맞는데 비교·계산·연결이 틀림<br>
&nbsp;mixed : 둘 이상이 섞여 하나로 정하기 어려움<br>
<b>near_miss</b> : 정답 후보를 2개까지 좁혔다가 틀렸으면 1<br>
<b>판단 요령</b> : 모델 출력에서 이미지를 서술한 문장을 먼저 찾아 실제 이미지와 비교한다.
서술이 틀렸으면 perception/ocr, 서술은 맞는데 결론이 틀렸으면 knowledge/reasoning.
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--failures", default="failures_to_tag_v2.csv")
    p.add_argument("--subjects", nargs="+",
                   default=["Biology", "Diagnostics_and_Laboratory_Medicine", "Agriculture"])
    p.add_argument("--per_subject", type=int, default=10)
    p.add_argument("--out_dir", default="analysis_out")
    p.add_argument("--image_dir", default="images", help="out_dir 기준 이미지 폴더 경로")
    args = p.parse_args()

    with open(args.failures, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f)
                if r["subject"] in args.subjects and r.get("manual_tag_needed") == "1"]

    picked = []
    for s in args.subjects:
        picked += sorted([r for r in rows if r["subject"] == s], key=lambda r: r["id"])[:args.per_subject]
    print(f"검토 대상 {len(picked)}개")

    os.makedirs(args.out_dir, exist_ok=True)

    # 태그 CSV
    tag_path = os.path.join(args.out_dir, "review_tags.csv")
    with open(tag_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["no", "subject", "id", "img_type", "gold", "pred",
                    "failure_type", "near_miss", "notes"])
        for i, r in enumerate(picked, 1):
            w.writerow([i, r["subject"], r["id"], r["img_type"], r["gold"], r["pred"], "", "", ""])

    # HTML 검토 페이지
    parts = [
        "<html><head><meta charset='utf-8'><title>오답 검토</title><style>"
        "body{font-family:sans-serif;max-width:1100px;margin:auto;padding:16px}"
        ".card{border:1px solid #ccc;border-radius:8px;padding:12px;margin:18px 0}"
        ".imgs img{max-width:100%;max-height:520px;border:1px solid #ddd;margin:4px}"
        "pre{white-space:pre-wrap;background:#f6f6f6;padding:8px;max-height:360px;overflow:auto}"
        ".ans{font-weight:bold}.guide{background:#fffbe6;padding:10px;border-radius:6px}"
        "</style></head><body>",
        f"<h2>오답 수동 태깅 ({len(picked)}개)</h2><div class='guide'>{GUIDE}</div>",
    ]
    for i, r in enumerate(picked, 1):
        imgs = "".join(
            f"<img src='{html.escape(os.path.join(args.image_dir, fn))}'>"
            for fn in r["images"].split("|") if fn
        )
        parts.append(
            f"<div class='card'><h3>#{i} {html.escape(r['id'])} "
            f"<small>({html.escape(r['img_type'])})</small></h3>"
            f"<div class='imgs'>{imgs}</div>"
            f"<p><b>문제</b><br>{html.escape(r['question'])}</p>"
            f"<p><b>선택지</b> {html.escape(r['options'])}</p>"
            f"<p class='ans'>정답: {html.escape(r['gold'])} &nbsp;|&nbsp; 모델 답: {html.escape(r['pred'])}</p>"
            f"<p><b>모델 출력</b></p><pre>{html.escape(r['raw_output'])}</pre></div>"
        )
    parts.append("</body></html>")
    html_path = os.path.join(args.out_dir, "review.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

    print(f"완료: {html_path} (브라우저로 열기), {tag_path} (엑셀로 태그 입력)")


if __name__ == "__main__":
    main()
