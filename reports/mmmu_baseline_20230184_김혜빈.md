# MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct

- 팀명: 3조
- 팀원: 김혜빈
- 작성일: 2026-09-27
- 재현 커맨드: `bash scripts/run_mmmu_eval_20230184_김혜빈.sh`

## 1. 환경 / 재현성

| 항목 | 값 |
|---|---|
| 모델 checkpoint | `Qwen/Qwen3-VL-4B-Instruct` (`ebb281ec70b05090aa6165b016eac8ec08e71b17`) |
| dtype | bf16 (`dtype="auto"`, 별도 양자화/캐스팅 없음 — 최신 transformers에서 `torch_dtype` 인자가 `dtype`으로 개명되었으나 동작은 동일) |
| 데이터셋 | `MMMU/MMMU` (`98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`), `validation` split, 30개 과목 config 전부 로드 (서브샘플링 없음, 총 900문제) |
| 추론 백엔드 | `transformers`의 `model.generate()` (vLLM 등 배치 서빙 프레임워크 없이 순정 API 사용 — 4B급 단일 GPU 추론이라 배치 최적화 없이도 완주 가능한 시간 내에 끝날 것으로 판단해 파이프라인 단순성을 우선함) |
| 사용 GPU | Google Colab, NVIDIA A100-SXM4-80GB (80GB) |
| 실측 peak VRAM (GB) | 9.76 GB (세션 누적 기준, `torch.cuda.max_memory_allocated()`) |
| 총 소요 시간 (900문제 기준) | 약 104.4분 (10문제 타이밍 측정: 문제당 평균 6.96초 × 900문제로 추정) |
| 의존성 | `pip install -q -U "git+https://github.com/huggingface/transformers"` <br> `pip install -q -U accelerate datasets qwen-vl-utils pillow` |
| 실행 커맨드 | `bash scripts/run_mmmu_eval_20230184_김혜빈.sh [model_path] [data_root] [output_path]` (인자 생략 시 기본값: `Qwen/Qwen3-VL-4B-Instruct` / `MMMU/MMMU` / `results/mmmu_baseline_20230184_김혜빈.jsonl`; revision은 스크립트 내부에 고정값으로 pin됨) |

## 2. 프롬프트

**객관식(multiple-choice) 문항**
```
{question}

(A) {option_A}
(B) {option_B}
...

Answer with the option's letter from the given choices directly.
```

**주관식(open-ended) 문항**
```
{question}

Answer the question using a single word or phrase.
```

- 출처: [MMMU-Benchmark/MMMU 공식 evaluation repo](https://github.com/MMMU-Benchmark/MMMU) — `configs/llava1.5.yaml`의 `multi_choice_example_format` / `short_ans_example_format`을 그대로 사용
- 선택 이유: 자체 설계 대신 MMMU 공식 평가 프레임워크가 사용하는 템플릿을 그대로 채용하여, 프롬프트 차이로 인한 채점 왜곡을 최소화하고 공식 수치(67.4)와의 비교 타당성을 높이기 위함
- 이미지가 여러 장인 문항은 질문 텍스트 내 `<image N>` placeholder 등장 순서대로 이미지를 나열해 함께 입력

## 3. 생성(Decoding) 설정

### 3.1 Sampling recipe

| 파라미터 | 값 |
|---|---|
| do_sample | True |
| temperature | 0.7 |
| top_p | 0.8 |
| top_k | 20 |
| repetition_penalty | 1.0 |
| presence_penalty | 0.0 (모델 repo 기본값, 별도 명시 없음) |
| seed | 42 (재현성 확보 목적으로 직접 고정 — 모델 repo에는 seed가 명시되어 있지 않음) |

- 출처: [`Qwen/Qwen3-VL-4B-Instruct`의 `generation_config.json`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/ebb281ec70b05090aa6165b016eac8ec08e71b17/generation_config.json) (해당 revision, 모델 제공사가 배포한 값). 임의로 정하지 않고 모델 repo에 실제로 포함된 생성 설정을 그대로 사용함.

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
|---|---|
| max_new_tokens | 128 (1차 실행) |
| 이미지 해상도 처리 | processor 기본값 그대로 사용 (`min_pixels`/`max_pixels` 별도 지정 없음) |

- 선택 근거: 900문제 전체를 시간 내에 완주하기 위해 초기값을 128로 짧게 설정. 그러나 5절 결과 분석 과정에서 계산/기술 계열 과목의 응답이 128 토큰 제한에 걸려 최종 답을 말하기 전에 잘리는 패턴을 발견했고, 저정확도 과목 2개(Architecture_and_Engineering, Energy_and_Power)를 `max_new_tokens=512`로 재실행해 정확도 개선을 확인함 (7절 참고). 즉 128은 "속도 우선" 트레이드오프였고, 그 대가(정확도 손실)를 실측으로 확인함.

## 4. 채점(파싱) 방식

- 사용한 파서/로직:
  - 객관식: MMMU 공식 repo `eval_utils.py`의 `parse_multi_choice_response`를 그대로 이식
  - 주관식: 동일 repo의 `parse_open_response` + `eval_open` (내부적으로 `normalize_str`, `check_is_number`, `extract_numbers` 사용) 그대로 이식
  - 출처: [MMMU-Benchmark/MMMU `eval_utils.py`](https://github.com/MMMU-Benchmark/MMMU/blob/main/mmmu/utils/eval_utils.py)
- 동작 방식 요약:
  - **객관식**: 응답에서 `(A)` 형태 → ` A ` 공백 형태 → 옵션 텍스트 포함 여부 순으로 후보를 탐색. 후보가 여러 개면 응답 내에서 가장 뒤에 등장한 것을 채택. 후보가 전혀 없으면 `random.choice`로 무작위 fallback (이 fallback 비율이 높은 과목일수록 실제 정확도도 낮게 나타남 — 7절 참고).
  - **주관식**: 응답을 문장 단위로 분리한 뒤 `"therefore"`, `"answer"`, `"final"` 등 지시어 뒤의 가장 짧은 구절과, 응답에서 추출한 숫자를 후보 목록으로 만들고, 정답(들)을 정규화(소문자화/숫자 반올림)한 뒤 포함 여부로 비교.

## 5. 결과

| No. | Subject | Data Num | Acc |
|---|---|---|---|
| 1 | Accounting | 30 | 0.300 |
| 2 | Agriculture | 30 | 0.567 |
| 3 | Architecture_and_Engineering | 30 | 0.133 |
| 4 | Art | 30 | 0.633 |
| 5 | Art_Theory | 30 | 0.800 |
| 6 | Basic_Medical_Science | 30 | 0.733 |
| 7 | Biology | 30 | 0.467 |
| 8 | Chemistry | 30 | 0.400 |
| 9 | Clinical_Medicine | 30 | 0.600 |
| 10 | Computer_Science | 30 | 0.533 |
| 11 | Design | 30 | 0.733 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 0.400 |
| 13 | Economics | 30 | 0.533 |
| 14 | Electronics | 30 | 0.400 |
| 15 | Energy_and_Power | 30 | 0.167 |
| 16 | Finance | 30 | 0.300 |
| 17 | Geography | 30 | 0.467 |
| 18 | History | 30 | 0.700 |
| 19 | Literature | 30 | 0.767 |
| 20 | Manage | 30 | 0.267 |
| 21 | Marketing | 30 | 0.533 |
| 22 | Materials | 30 | 0.300 |
| 23 | Math | 30 | 0.367 |
| 24 | Mechanical_Engineering | 30 | 0.133 |
| 25 | Music | 30 | 0.300 |
| 26 | Pharmacy | 30 | 0.567 |
| 27 | Physics | 30 | 0.500 |
| 28 | Psychology | 30 | 0.733 |
| 29 | Public_Health | 30 | 0.367 |
| 30 | Sociology | 30 | 0.667 |
| | **Overall (macro avg)** | **900** | **0.4789** |

계산식: Overall = mean(30개 과목 accuracy) (과목당 문제 수가 정확히 30개로 균등하므로, 전체 900문제 중 맞힌 개수의 비율과 수학적으로 동일함)

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---|
| 공식 (Qwen3-VL Technical Report) | 67.4 |
| 우리 재현 결과 | 47.89 |
| 차이 (Δ) | -19.51%p |

## 7. 격차 분석

측정한 종합 점수(macro avg 47.89%)는 공식 수치(67.4%)보다 19.5%p 낮다. 평균 응답 길이가 200자를 넘는 과목의 평균 정확도는 33.0%인 반면, 응답이 짧은(100자 미만) 과목은 평균 60.9%였다. 응답이 긴 과목일수록 파싱 단계에서 답 letter를 명확히 찾아낸 비율도 낮았다(긴 과목 57.0% vs 짧은 과목 98%+). 이는 `max_new_tokens=128` 제한 하에서 계산/기술 계열 문제일수록 모델이 풀이 과정을 서술하다 최종 답을 말하기 전에 응답이 잘렸을 가능성을 시사했다. 이를 확인하기 위해 저정확도 과목 2개(Architecture_and_Engineering, Energy_and_Power)를 `max_new_tokens=512`로 재실행한 결과, 각각 0.133→0.200, 0.167→0.367로 정확도가 개선되어 가설이 뒷받침됐다. 반례로 Music은 응답이 매우 짧고(5.5자) letter 검출률도 100%였지만 정확도는 30%에 그쳐, 이는 모델의 악보 이미지 이해 한계로 보인다. 즉 격차의 상당 부분은 생성 예산 설정에서 기인하고, 일부는 모델의 실제 능력 한계다.

## 8. 기타 특이사항 / 한계 (Optional)

- MMMU 문항 중 일부(예: Architecture_and_Engineering의 특정 문항)는 객관식이 아닌 주관식(open-ended)이었음. 초기 파이프라인이 이를 고려하지 않아 실행 중 에러가 발생했고, MMMU 공식 repo의 open-ended 채점 로직(`parse_open_response`/`eval_open`)을 추가로 이식해 대응함.
- `do_sample=True`(비결정적 샘플링)를 사용했기 때문에, seed를 고정했더라도 GPU/라이브러리 버전에 따라 재실행 시 점수가 소폭 달라질 수 있음.
- Colab 런타임이 중간에 재연결되면서 체크포인트(JSONL append) 방식 덕분에 데이터 유실 없이 이어서 실행할 수 있었음.
- 다음에 시도해보고 싶은 것: 전체 900문제를 `max_new_tokens=512`로 재실행해 격차가 얼마나 더 줄어드는지 정량적으로 확인, vLLM 도입을 통한 속도 개선.
