# MMDL: Qwen3-VL-4B performance improvement

멀티모달 딥러닝 3조의 평가, LoRA 학습, 실험 설정과 결과를 보관한다.
최종 과제 모델은 단일 R20이다. 3모델·5모델 투표는 추가 탐구다.

## 폴더 구성

```text
MMDL/
  assignment/       과제 1 기록과 최종 프로젝트 설명
  code/             평가·학습·실험 코드
    qwenvl/         학습에 필요한 Qwen Python 패키지
  results/          기존 과제 1 결과, 최종 집계와 학습 설정
  README.md
  requirements.txt
  requirements_training.txt
  requirements_evaluation.txt
  THIRD_PARTY_NOTICES.md
```

모델 가중치, 원본 Parquet·이미지, 학습 JSON, 가상환경, 체크포인트와 백업은 Git에 넣지 않는다.
`results/results_single_run/`은 이 저장소에 이미 올라가 있던 과제 1 결과를 보존한 것이다.
이번 추가분은 문항별 새 정답·출력 대신 집계 결과만 포함한다.

## 먼저 알아둘 평가 한계

현재 보관한 `code/eval_mmmu_vllm.py`는 과거 점수를 계산한 원본이다.
질문 본문의 이미지 참조만 수집해 선택지 그림이 빠지는 문제가 있다.
전체 Test 366문항이 영향을 받는 코드 경로이며 271문항은 이미지가 전달되지 않는다.
기존 50.60%와 51.70%는 이 고정 파이프라인 기준이다.
수정 후 재평가를 완료했다고 주장하지 않는다. 과거 코드의 동작을 몰래 바꾸지 않았다.

## 주요 코드

| 파일 | 역할 |
|---|---|
| `code/train.py` | 실제 R20 학습에 사용한 entry snapshot |
| `code/run_train.py` | 저장된 학습 인자를 읽고 경로를 지정해 위 entry를 실행 |
| `code/qwenvl/` | 학습 모델·데이터 전처리·Trainer 관련 소스 |
| `code/eval_mmmu_vllm.py` | 고정 MMMU Validation 평가 원본 |
| `code/run_full_test_base_R20.py` | 전체 Test 순정·R20 평가 당시의 러너 |
| `code/run_mmmu_pro_final.py` | MMMU-Pro 최종 후보 평가 러너 |
| `code/frozen_ensemble_rule.py` | 고정 투표 규칙 |
| `code/01_*`부터 `code/09_*` | 초기 실패 진단과 보강 실험 |
| `code/run_*_experiment.py`, `code/*_server.py` | 당시 실험·학교 서버용 스냅샷 |

## 환경

학습: 학교 RTX 4090, Python 3.12, torch 2.6.0+cu126,
transformers 4.57.1, peft 0.17.1, accelerate 1.7.0,
flash-attn 2.7.4.post1.

평가: 개인 RTX 4080 SUPER, WSL2, Python 3.13,
torch 2.8.0, transformers 4.57.6, vLLM 0.11.0.

실제 환경의 패키지 목록은 `requirements_training.txt`와
`requirements_evaluation.txt`에 보관했다. 두 목록은 관측한 환경 전체이며
두 환경을 하나로 섞거나 그대로 모든 GPU에 설치하는 용도가 아니다.
CUDA와 PyTorch를 해당 장비에 맞춰 먼저 구성해야 한다.

## 모델과 데이터 준비

원본 모델은 `Qwen/Qwen3-VL-4B-Instruct`, revision
`ebb281ec70b05090aa6165b016eac8ec08e71b17`이다.
모델과 학습 데이터는 별도로 다운로드하거나 학교 서버의 검증된 파일을 준비한다.

R20은 `train.json` 4,000개와 `internal_dev.json` 400개 및 상대 경로의 이미지를 사용한다.
형식은 `image`와 human/gpt `conversations`를 가진 Qwen 학습 JSON이다.
텍스트 Replay 문항에는 image가 없다. 분할·검토·중복 검사 기록을 함께 보관해야 한다.
원본 revision과 최종 학습 비율은 `results/experiment_settings.json`에 있다.

이 Git 묶음에는 원본 데이터나 이미지, 모델 가중치와 최종 adapter가 없다.
2026-11-24 학교 서버 수집본에서 확대 Dev 평가와 추가 실험 보조 코드를 확보했다.
다만 초기 후보 생성·그룹 분할·파일럿·비율별 데이터 생성·텍스트 Replay 후보 생성의 일부 코드는
이 압축에서도 독립 실행 파일로 확인되지 않았다.
따라서 Git clone만으로 데이터 선별부터 모든 과거 실험이 자동 재현된다고 주장하지 않는다.
남은 초기 데이터 생성 코드 또는 동일한 준비 데이터·분할·선별 기록의 별도 확보가 필요하다.

## R20 학습 실행

Linux 또는 학교 컨테이너의 학습 환경에서 실행한다.
다음 경로는 본인이 준비한 모델·데이터·출력 위치로 바꾼다.

```bash
python code/run_train.py --model /shared_dir/mmdl/models/Qwen3-VL-4B-Instruct --data /shared_dir/mmdl/checkpoints/replay_compare_20261119_214637_363626/data/R20 --output /shared_dir/mmdl/checkpoints/r20_reproduction
```

실제 학습을 시작하지 않고 명령만 확인하려면 `--dry-run`을 추가한다.
기본 인자는 `results/r20_training_arguments.txt`의 실제 기록에서 읽고
모델·데이터·출력 경로만 지정한다. GPU 학습을 이 업로드 준비 과정에서 실행하지 않았다.

## Validation 평가 실행

LoRA를 병합한 모델 폴더나 순정 모델 경로를 지정한다.
이 명령은 과거 Validation 파이프라인의 실행이며 위의 이미지 입력 한계가 있다.

```bash
python code/eval_mmmu_vllm.py --model_path /path/to/model --data_root MMMU/MMMU --out_dir /path/to/evaluation_output
```

## 전체 Test와 MMMU-Pro 러너

`run_full_test_base_R20.py`와 `run_mmmu_pro_final.py`는 실행 당시 경로를 보존한 스냅샷이다.
이 러너들은 원래 `/mnt/c/Users/K/Desktop/MMDL` 및 그 아래 `scripts/`,
`evaluation_reference/`, `merged_models/`, `results/` 구조를 참조한다.
여기서는 폴더를 단순화해 스크립트를 `code/`에 모았다.
다른 컴퓨터에서는 `PROJECT`, 모델 경로와 보조 파일 경로를 맞춘 후 실행해야 한다.
`code/`를 실행용 프로젝트의 `scripts/` 위치에 복사하는 방식도 가능하다.
실행 전 저장된 revision과 코드 SHA 검사 조건을 확인하고, 코드 변경과 새 결과를 별도 기록한다.

모델·원본 데이터·고정 입력 manifest·정답 파일은 별도로 준비해야 한다.
정답은 추론 입력이나 문항별 모델 선택에 사용하지 않는다.
MMMU-Pro standard와 vision은 같은 1,730문항의 다른 입력 형식이다.

## 결과

| 평가 | 순정 | R20 |
|---|---:|---:|
| MMMU Test 10,500개, 기존 파이프라인 | 5,313 / 50.60% | 5,428 / 51.70% |
| MMMU-Pro standard | 624 / 36.07% | 642 / 37.11% |
| MMMU-Pro vision | 496 / 28.67% | 514 / 29.71% |

집계는 `results/*_summary.json`, 설정은 `results/experiment_settings.json`에 있다.
입력 누락, 단일 시드와 반복 개발 탐색의 한계를 함께 보고한다.

## 팀 협업

팀장 계정의 공개 저장소 하나를 사용하고 모든 팀원을 collaborator로 등록한다.
각자 본인 계정으로 commit과 pull request를 작성한다.
최종 변경은 pull request 검토 후 기본 브랜치 `main`으로 merge한다.
다른 사람의 기여 이력을 대신 만들지 않는다.

## 출처

- Qwen3-VL: https://github.com/QwenLM/Qwen3-VL
- MMMU: https://github.com/MMMU-Benchmark/MMMU
- MMMU-Pro: https://huggingface.co/datasets/MMMU/MMMU_Pro
- ScienceQA: https://github.com/lupantech/ScienceQA
- CoSyn-400K: https://huggingface.co/datasets/allenai/CoSyn-400K

학습 소스 출처와 라이선스는 `THIRD_PARTY_NOTICES.md`를 확인한다.

## 학교 서버에서 보완한 실험 코드

`code/server_*.py`는 학교 서버에서 실제 사용한 추가 실험 코드의 원본 보관본이다.
확대 Dev 평가, 학습률과 rank 비교, LoRA 배율 비교, 텍스트 Replay 진단,
공학 문항 검증·분할·오염 검사·원출처 대조 및 조합 데이터 준비 코드를 포함한다.
기존 Git 파일과 내용이 완전히 같은 파일은 중복 저장하지 않았다.
원래 서버 상대 경로와 Git 파일 경로, SHA256은 `results/server_source_manifest.json`에 있다.

이 코드들은 당시 작업 폴더와 상대 경로를 참조한다.
`code/`에서 각 파일을 바로 실행하는 방식으로 전체 과거 실험이 재현되는 것은 아니다.
원래 폴더 배치가 필요하면 아래 도구로 복원 대상을 먼저 확인한다.

```bash
python code/restore_server_sources.py --root /shared_dir/mmdl
```

기본 실행은 파일을 쓰지 않는다. 출력 경로와 기존 파일 충돌을 검토한 뒤
`--apply`를 추가하면 원래 상대 경로에 코드만 복사한다.
다른 내용의 기존 파일이 있으면 덮어쓰지 않고 중단한다.
복원은 모델·데이터 준비나 학습 실행을 수행하지 않는다.
최종 R20 학습은 위의 `run_train.py` 안내를 우선 사용한다.
