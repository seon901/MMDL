# MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct

- **팀명**: 3조(밥메이트)
- **팀원**: 강인선, 김혜빈, 이현주
- **작성일**: 2026.09.27
- **재현 커맨드**: `python3 scripts/eval_mmmu_vllm.py --model_path Qwen/Qwen3-VL-4B-Instruct --data_root MMMU/MMMU` (WSL2 환경, 저장소 루트에서 실행)

---

## 1. 환경 / 재현성

| 항목 | 값 |
|---|---|
| 모델 checkpoint | `Qwen/Qwen3-VL-4B-Instruct` (ebb281ec70b05090aa6165b016eac8ec08e71b17) |
| 추론 백엔드 | vLLM 0.11.0 (WSL2, Ubuntu). 전환 배경은 8번 섹션에 기술한다. |
| 사용 GPU | NVIDIA GeForce RTX 4080 SUPER 16GB, WSL2 위 CUDA 12.8 toolkit + torch 2.8.0+cu128 |
| 실측 peak VRAM | 약 15.5/16GB |
| 총 소요 시간 | 47.0분 (900문제 전체를 처음부터 한 번에 실행, 모델 로딩 시간 제외) |
| 의존성 | [requirements.txt](../requirements.txt) (pip freeze 전체 목록). 핵심 버전은 Python 3.13.13, vllm 0.11.0, torch 2.8.0+cu128, transformers 4.57.6, qwen-vl-utils 0.0.14, datasets 5.0.1, huggingface_hub 0.36.2, numpy 2.2.6이다. |
| 실행 커맨드 | `python3 scripts/eval_mmmu_vllm.py --model_path Qwen/Qwen3-VL-4B-Instruct --data_root MMMU/MMMU` `--model_path`에는 HF repo id 또는 로컬 checkpoint 경로를, `--data_root`에는 HF repo id 또는 로컬 MMMU 데이터 경로를 줄 수 있다. 실행 전에 CUDA 12.8 toolkit이 잡혀 있어야 하므로 `export CUDA_HOME=/usr/local/cuda-12.8`, `export PATH=$CUDA_HOME/bin:$PATH`, `export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH`를 먼저 실행한다. |

## 2. 프롬프트

**MCQ 문제 프롬프트 (1차 실행)**:

```
Question: {question}
Options:
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}
Answer the question. At the very end of your response, write your final
choice on a new line in exactly this format:
Answer: X
where X is a single option letter.
```

**MCQ 문제 프롬프트 (재시도)**: 1차 실행에서 답을 추출하지 못한 MCQ에만 사용한다.

```
Question: {question}
Options:
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}
First write your final choice on its own line in exactly this format:
Answer: X
where X is a single option letter. Then, after that line, briefly explain
your reasoning.
```

**개방형(open-ended) 문제 프롬프트**: 선택지가 없는 문제에는 아래 별도 프롬프트를 사용한다. 재시도하지 않는다.

```
Question: {question}
Answer the question. At the very end of your response, write your final
answer on a new line in exactly this format:
Answer: <your final answer>
Keep the final answer a short phrase or number, without units or explanation
unless the question specifically asks for them.
```

스크립트는 과목마다 이렇게 동작한다. 먼저 전체 문제를 1차 프롬프트로 실행한다. 그다음 답을 추출하지 못한 MCQ만 골라 재시도 프롬프트로 다시 실행한다. 재시도 대상은 정답 여부가 아니라 답 추출 실패 여부만으로 정한다. 채점은 재시도까지 끝난 최종 예측에 대해 한 번만 한다.

- **출처**: 직접 설계했다. 프롬프트 구조는 MMMU 공식 저장소와 Qwen이 벤치마크 재현에 사용한 lmms-eval이 공통으로 채택한 관례, 즉 답을 명시적으로 표시하고 파서가 이를 우선 탐색하는 방식을 참고했다. 원문 코드를 그대로 가져온 것은 아니다.
- **선택 이유**: 초기에는 letter만 답하라는 지시를 사용했다. Accounting 30문제로 시험하니 max_new_tokens=64에서 답을 추출하지 못한 문제가 8개, 정답이 10개였다. 마지막 줄에 "Answer: X"를 쓰게 하는 지금의 프롬프트로 바꾸자 모델이 풀이를 길게 쓰게 되어 문제당 평균 시간이 0.93초에서 9.8초(256토큰)로 늘었고, 256토큰에서는 답 추출 실패가 16개로 늘고 정답은 9개였다. 512토큰으로 늘리자 실패 10개, 정답 16개가 되어 이 프롬프트와 512토큰을 채택했다. 한 과목 30문제로 한 사전 실험이라 표본이 작다. 512 토큰으로도 남은 파싱 실패는 모델이 풀이를 길게 서술하다 답까지 도달하지 못한 경우였다. 그래서 답을 추출하지 못한 MCQ에 한해 답을 먼저 쓰게 하는 프롬프트로 재시도하도록 했다. 선택지가 없는 개방형 문제는 letter 대신 "Answer: <답>" 형식의 별도 프롬프트를 사용하며 재시도하지 않는다.

## 3. 생성(Decoding) 설정

### 3.1 Sampling recipe

| 파라미터 | 값 |
|---|---|
| `do_sample` | `True` |
| `temperature` | `0.7` |
| `top_p` | `0.8` |
| `top_k` | `20` |
| `repetition_penalty` | `1.0` |
| `presence_penalty` | `1.5` |
| `seed` | `3407` |

- **출처**: HuggingFace 모델 카드 `Qwen/Qwen3-VL-4B-Instruct`의 "Generation Hyperparameters – VL" 섹션이다. 동일 값이 Qwen 공식 GitHub 저장소 `QwenLM/Qwen3-VL`의 "Evaluation Reproduction" 섹션에도 기재되어 있다.
- 최초 시도에서는 transformers를 백엔드로 사용해 `presence_penalty`를 적용하지 못했다. transformers.generate()는 이 파라미터를 지원하지 않기 때문이다. vLLM으로 전환한 뒤 공식 recipe를 값 그대로 완전히 적용했다.

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
|---|---|
| `max_new_tokens` | `512` |
| 이미지 해상도 처리 | `min_pixels=256*28*28`, `max_pixels=1024*28*28` |
| `max_model_len` (vLLM) | `4096` |
| `max_num_seqs` (vLLM) | `4` |

**선택 근거**

`max_new_tokens`는 공식 recipe의 out_seq_length(16384)와 RTX 4090 기준 강의 권장값(2048)보다 작은 512로 정했다. Accounting 30문제 사전 실험(transformers 기준)에서 256토큰은 답 추출 실패 16개와 정답 9개, 512토큰은 실패 10개와 정답 16개였고 문제당 평균 시간은 9.8초에서 17.6초로 늘었다. 512보다 큰 값은 시간 때문에 시험하지 않았다. 그 결과 900문제 실행에서도 MCQ 142개가 512토큰 안에 답을 내지 못해 재시도가 필요했다. 이 선택은 성능을 제한하는 요인일 수 있어 7번에서 원인 후보로 다뤘다.

이미지 해상도는 RTX 4080 Super 16GB 기준 초기 실행에서 VRAM이 15.4GB까지 근접하고 원본 546만 픽셀 이미지가 포함되어 있음을 확인해 제한을 적용했다. 강의 권장값(max_pixels 5120×28×28)보다 작게 잡은 셈인데, 이미지 토큰 1개가 약 32×32 픽셀에 해당하므로(측정: 약 77만 픽셀에서 이미지 토큰 750개) 권장값대로 하면 이미지 한 장이 약 3,900토큰이 되어 max_model_len 4096 안에 질문과 답변이 들어가지 않는다. max_model_len은 아래처럼 KV 캐시 여유 때문에 4096으로 정했다.

vLLM 전환 초기에는 `max_model_len=8192`로 설정했으나 KV 캐시 여유가 부족해 엔진 초기화 오류가 발생했다. `max_model_len`을 4096으로 낮추고 `gpu_memory_utilization`을 0.90으로 올려 해결했다. 그 결과 이미지가 많거나 응답이 긴 과목에서 동시 처리 가능한 요청 수가 지나치게 줄어 처리 속도가 급격히 떨어지는 부작용이 생겼다. `max_num_seqs=4`로 동시 처리 요청 수를 명시적으로 제한해 해결했다. 이 설정은 모델이 생성하는 답변 내용에는 영향을 주지 않는 순수 스케줄링 파라미터다.

## 4. 채점(파싱) 방식

**MCQ/개방형 구분**: `options` 필드가 비어 있으면 개방형, 값이 있으면 MCQ로 판단한다.

**MCQ 문제**: 모델 응답에서 정규식 `Answer:\s*[*(\[]*\s*([letters])`로 명시적 "Answer: X" 줄을 우선 탐색한다. `Answer: **B**`나 `Answer: (B)`처럼 기호가 붙은 표기도 읽는다. 매칭되지 않으면 응답에 등장하는 선택지 letter 중 마지막으로 언급된 것을 채택한다. 둘 다 실패하면 답 없음으로 두고, MCQ이면 재시도 프롬프트로 다시 실행해 같은 방식으로 추출한다. 재시도 후에도 추출하지 못하면 오답으로 처리한다. `Answer: **B**`처럼 기호가 붙은 표기는 재시도하지 않은 MCQ 705개 중 2개뿐이었다. 재시도 대상은 정답 여부를 보지 않고 답 추출 실패 여부만으로 정하며, 채점은 재시도까지 끝난 최종 예측에 대해 한 번만 수행한다. letter pool은 A부터 P까지 확장했다. 선택지가 9개 이상인 문제에서 기존 A~H 범위로는 인덱스 오류가 발생함을 확인해 대응했다.

**개방형 문제**: "Answer:" 뒤에 오는 텍스트를 추출하고(여러 번 나오면 마지막 것을 채택) 정답과 비교한다. 먼저 정규화 매칭을 시도해 정답이 숫자면 여러 소수점 표기 변형을 만들어 비교하고 정답이 리스트면 그중 하나라도 일치하면 정답으로 처리한다. 정규화 매칭에 실패하면 문자열 부분 일치를 fallback으로 허용한다. 이 채점 로직은 MMMU 공식 평가 스크립트 `eval_utils.py`의 `eval_open` 함수 구조를 참고해 직접 재구현했다. 원본 코드를 그대로 가져온 것은 아니며 숫자 소수점 변형 처리 등 세부 구현은 자체적으로 추가했다.

## 5. 결과

| No. | Subject | Data Num | Acc |
|---|---|---|---|
| 1 | Accounting | 30 | 0.700 |
| 2 | Agriculture | 30 | 0.467 |
| 3 | Architecture_and_Engineering | 30 | 0.233 |
| 4 | Art | 30 | 0.667 |
| 5 | Art_Theory | 30 | 0.800 |
| 6 | Basic_Medical_Science | 30 | 0.667 |
| 7 | Biology | 30 | 0.267 |
| 8 | Chemistry | 30 | 0.400 |
| 9 | Clinical_Medicine | 30 | 0.567 |
| 10 | Computer_Science | 30 | 0.433 |
| 11 | Design | 30 | 0.833 |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 0.300 |
| 13 | Economics | 30 | 0.700 |
| 14 | Electronics | 30 | 0.233 |
| 15 | Energy_and_Power | 30 | 0.400 |
| 16 | Finance | 30 | 0.500 |
| 17 | Geography | 30 | 0.567 |
| 18 | History | 30 | 0.700 |
| 19 | Literature | 30 | 0.733 |
| 20 | Manage | 30 | 0.500 |
| 21 | Marketing | 30 | 0.800 |
| 22 | Materials | 30 | 0.433 |
| 23 | Math | 30 | 0.467 |
| 24 | Mechanical_Engineering | 30 | 0.367 |
| 25 | Music | 30 | 0.200 |
| 26 | Pharmacy | 30 | 0.567 |
| 27 | Physics | 30 | 0.567 |
| 28 | Psychology | 30 | 0.667 |
| 29 | Public_Health | 30 | 0.733 |
| 30 | Sociology | 30 | 0.533 |
| | **Overall (macro average)** | **900** | **0.533** |

계산식: `Overall = mean(30개 과목 accuracy)`. MMMU val은 과목당 정확히 30문제로 균등하므로 이는 전체 900문제 중 맞힌 개수의 비율 480/900과 수학적으로 동일하다. 이 표는 `scripts/eval_mmmu_vllm.py`를 처음부터 한 번 실행해 얻은 값이다.

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---|
| 공식 (Qwen3-VL Technical Report) | 67.4 |
| 우리 재현 결과 | 53.3 |
| 차이 (Δ) | -14.1%p |

## 7. 격차 분석

vLLM 전환 후 900문제를 처음부터 다시 실행한 최종 결과는 53.3%로, 공식 67.4%와 14.1%p 차이가 난다. 원인을 세 갈래로 분해했다.

첫째, 생성 예산과 파싱이다. MCQ 847문제 중 142문제(16.8%)가 1차에서 답을 추출하지 못했다. raw_output을 일부 확인하니 모델이 답을 몰라서가 아니라 풀이가 길어 512토큰 안에 답까지 도달하지 못한 경우가 많았다. 답을 먼저 쓰게 한 재시도로 124문제에서 답을 추출했고 54문제를 맞혔다. 재시도가 없었다면 정확도는 47.3%였을 것이다. 그래도 18문제는 답을 내지 못했다.

둘째, 개방형 문제다. 53문제 중 13문제만 맞혀 24.5%로 MCQ(55.1%)보다 크게 낮다. 27문제는 답을 추출하지 못했고 추출한 26문제 중 13문제가 정답이었으므로, 주된 원인은 채점보다 답을 내지 못한 것으로 보인다. 개방형은 재시도하지 않았다.

셋째, 과목별 편차다. Music(0.200)은 raw_output에서 악보의 음표와 박자표를 인식하는 단계부터 확신하지 못했다. Electronics(0.233)는 30문제 중 14문제가 개방형이다.

정리하면 개방형이 약 1.8%p, 끝내 답을 내지 못한 MCQ가 약 1.1%p를 차지하는 것으로 어림된다. 나머지 약 11%p는 설명하지 못했다. 512토큰과 낮은 이미지 해상도(max_pixels 1024×28×28)가 공식 recipe의 생성 길이(16384토큰)나 강의 권장값(2048토큰, 5120×28×28)보다 제한적이라는 점이 유력한 후보이나 검증하지 못했다. 또한 같은 방식의 두 실행이 56.0%와 53.3%로 2.7%p 달랐으므로 이 격차에는 그 정도의 측정 변동이 포함된다(8번).

## 8. 기타 특이사항 / 한계

**backend 전환**: 최초에는 transformers.generate()로 파이프라인을 완성해 900문제를 3.10시간에 처리했고 정확도 47.0%를 얻었다. 그러나 이 방식은 문제를 한 번에 하나씩만 처리해 속도가 느렸고, transformers.generate()가 presence_penalty 파라미터를 지원하지 않아 공식 sampling recipe를 완전히 적용할 수 없었으며, 개방형 문제 53개를 채점하지 않고 전부 오답으로 처리하는 한계가 있었다. 이 세 가지를 해결하기 위해 배치 처리와 공식 recipe를 온전히 지원하는 vLLM으로 전환했고, WSL2 환경에서 실행했다. 전환 후 처음 얻은 정확도는 49.8%였다. 이 실행은 중간에 설정을 바꿔 이어서 돌렸기 때문에 전체 소요 시간을 한 번에 측정하지 못했다. 이후 900문제를 처음부터 한 번에 다시 실행했을 때는 47.0분이 걸려, transformers의 3.10시간보다 약 4배 빠르다. 정확도는 47.0%에서 53.3%가 됐는데, 이 차이에는 presence_penalty 적용, 개방형 채점 추가, 프롬프트 변경이 함께 들어 있어 backend만의 효과는 아니다.

**파싱 실패 140문제 재시도**: vLLM 전환 직후 결과를 검토하는 과정에서 MCQ 847문제 중 140문제가 답을 추출하지 못해 파싱에 실패한 것을 발견했다. 실제 모델 응답(raw_output)을 확인해보니 모델이 답을 몰라서가 아니라 풀이를 길게 서술하다 512토큰 제한 안에 최종 답까지 도달하지 못한 경우가 대부분이었다. 이를 해결하기 위해 프롬프트에서 답을 마지막에 쓰게 하던 순서를 답을 먼저 쓰고 설명은 나중에 쓰는 순서로 바꿔, 파싱에 실패했던 이 140문제만 다시 실행했다. 나머지 760문제는 원래 결과를 그대로 유지하고 이 140문제의 새 결과만 병합해 정확도 56.0%를 얻었다(병합 실험). 다만 이 값은 서로 다른 두 프롬프트의 결과를 합친 것이라 스크립트 하나로는 그대로 재현되지 않는다. 그래서 이 재시도 과정을 별도 스크립트 대신 `eval_mmmu_vllm.py` 안에 넣어, 과목마다 1차 실행 후 답을 추출하지 못한 MCQ만 즉시 재시도한 뒤 채점하도록 했다. 이 스크립트를 처음부터 다시 실행한 결과를 최종 baseline으로 삼았다(5, 6번). 이 실행에서는 MCQ 142개가 1차에서 답을 추출하지 못해 재시도됐고 그중 124개에서 답을 추출했다. 최종 정확도는 53.3%로, 병합 실험의 56.0%보다 2.7%p 낮다.

**답 먼저 프롬프트의 전체 적용 미검증**: 답을 먼저 쓰게 하는 프롬프트를 1차 실행부터 900문제 전체에 적용하면 파싱 실패 자체가 줄어들 수 있다. 그러나 그렇게 전체를 돌려본 적이 없어 지금의 "답 마지막 1차 실행 + 답 먼저 재시도" 방식과 비교하지 못했다.

**개방형 문제 채점 로직 미검증**: 개방형 문제의 채점은 MMMU 공식 평가 스크립트의 eval_open 함수 구조를 참고해 직접 재구현한 것이며, 원본 코드와 나란히 놓고 결과가 정확히 일치하는지 대조한 적은 없다. 이번 실행에서 개방형 정확도는 24.5%(13/53)로 MCQ의 55.1%보다 크게 낮았다. 27문제는 답 자체를 추출하지 못해 채점 대상이 되지 못했고, 답을 추출한 26문제 중에서는 13문제가 정답이었다. 그래서 낮은 정확도는 채점 로직보다 답을 내지 못한 경우가 더 큰 원인으로 보이지만, 채점 로직이 정답을 놓친 경우가 있는지는 확인하지 못했다. 개방형 문제에는 재시도를 적용하지 않았고, 추출에 실패한 27문제에 재시도를 적용했을 때의 효과도 확인하지 못했다.

**실행 간 결과 변동**: 같은 코드와 시드로 실행해도 결과가 달라졌다. vLLM 전환 직후 실행과 비교했을 때 1차 MCQ 예측이 같은 비율은 89.9%(624/694)였다. 결과가 갈린 문제 중 이번 실행만 정답인 문제는 16개, 이전 실행만 정답인 문제는 31개로 한쪽에 쏠려 있었고(McNemar p=0.041) 원인은 확인하지 못했다. 이전 실행의 앞쪽 19개 과목은 `max_num_seqs`를 지정하지 않고 실행해 동시 처리 설정도 달랐다. 병합 실험(56.0%)과 이번 실행(53.3%)의 전체 정확도도 2.7%p 차이가 났다. 따라서 이 baseline은 몇 %p의 변동을 포함하는 값으로 읽어야 하고, 이후 fine-tuning 전후를 비교할 때도 이 정도보다 큰 차이여야 개선으로 볼 수 있다.

**flash_attention_2 미적용**: Qwen3-VL 모델 카드는 transformers로 모델을 불러올 때 flash_attention_2를 적용하면 속도와 메모리 사용이 개선된다고 안내한다. 이를 적용해보려 했으나 Windows에서 flash-attn을 설치하려면 비공식 커뮤니티 wheel을 써야 했고, 그 wheel이 빌드된 CUDA 버전이 당시 설치된 torch와 맞지 않아 재설치가 필요한 상황이었다. 이후 vLLM으로 전환하면서 이 시도는 중단했다. 다만 vLLM 자체가 Flash Attention backend를 기본으로 사용하기 때문에, 최종 결과에는 이 최적화가 이미 반영되어 있다.

**실행 커맨드 파라미터화**: 초기 버전은 모델 checkpoint 경로와 데이터셋 revision이 코드 상단에 고정되어 있어 재현 시 코드를 직접 열어 값을 바꿔야 했다. argparse를 추가해 `--model_path`, `--data_root` 등을 커맨드라인 인자로 받도록 수정했으며, 인자를 주지 않으면 기존과 동일한 기본값(HuggingFace repo id, pin된 revision)으로 동작해 재현성을 유지한다.

**로컬 환경의 제약**: 사용한 GPU가 RTX 4080 Super 16GB라서, vLLM의 KV 캐시 여유가 넉넉하지 않았다. 이 때문에 `max_model_len`을 4096으로, 동시 처리 개수(`max_num_seqs`)를 4로 제한해야 했는데, 더 큰 VRAM을 가진 환경이었다면 이런 제약 없이 더 길고 많은 요청을 동시에 처리할 수 있었을 것이다. 또한 vLLM은 Windows에서 네이티브로 동작하지 않아 WSL2를 거쳐야 했고, 그 과정에서 WSL의 CUDA toolkit 버전을 GPU 드라이버와 새로 설치한 torch에 맞춰 별도로 설정해야 했다. 순수 Linux 서버 환경이었다면 이런 절차 없이 곧바로 vLLM을 설치해 사용할 수 있었을 것이다.
