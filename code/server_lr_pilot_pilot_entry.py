import copy
import json
import math
import os
import sys
from pathlib import Path

import torch

repo = Path(os.environ["PILOT_REPO"])
data = Path(os.environ["PILOT_DATA"])
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "qwenvl/train"))

import peft
import qwenvl.data as registry
import train_qwen

for name, filename in (
    ("pilot_train", "train.json"),
    ("pilot_dev", "internal_dev.json"),
):
    registry.data_dict[name] = {
        "annotation_path": str(data / filename),
        "data_path": str(data),
    }

original_get_peft_model = peft.get_peft_model

def checked_get_peft_model(*args, **kwargs):
    model = original_get_peft_model(*args, **kwargs)
    names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    assert names, "학습 가능한 파라미터가 없습니다."
    assert all("lora_" in name for name in names), \
        "LoRA 이외의 학습 파라미터가 있습니다."
    assert all("language_model" in name for name in names), \
        "언어 모델 이외의 학습 파라미터가 있습니다."
    assert not any("visual" in name for name in names), \
        "시각 모델에 학습 파라미터가 있습니다."

    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"학습 파라미터 수: {trainable:,}", flush=True)
    print("시각 인코더와 병합 모듈 동결 확인", flush=True)
    return model

peft.get_peft_model = checked_get_peft_model

original_make_data = train_qwen.make_supervised_data_module

def make_pilot_data(processor, data_args):
    train_module = original_make_data(
        processor,
        data_args=data_args,
    )

    dev_args = copy.deepcopy(data_args)
    dev_args.dataset_use = "pilot_dev"
    dev_module = original_make_data(
        processor,
        data_args=dev_args,
    )

    train_module["eval_dataset"] = dev_module["train_dataset"]

    assert len(train_module["train_dataset"]) == 320, \
        "학습 데이터가 320개가 아닙니다."
    assert len(train_module["eval_dataset"]) == 80, \
        "개발 데이터가 80개가 아닙니다."

    if os.environ.get("CHECK_DATA") == "1":
        from qwenvl.data.data_processor import preprocess_qwen_visual

        for split, dataset, filename in (
            ("학습", train_module["train_dataset"], "train.json"),
            ("개발", train_module["eval_dataset"], "internal_dev.json"),
        ):
            rows = json.loads(
                (data / filename).read_text(encoding="utf-8")
            )
            maximum = 0

            for index, row in enumerate(rows):
                source = dict(row)
                source["data_path"] = str(data)

                sample = preprocess_qwen_visual(
                    [source],
                    dataset.processor,
                )
                length = sample["input_ids"].shape[-1]
                labels = sample["labels"].reshape(-1)

                assert length <= 2048, (
                    f"{split} {index}번이 길이 제한을 초과했습니다: "
                    f"{length}토큰"
                )
                assert int((labels != -100).sum()) > 0, (
                    f"{split} {index}번에 정답 토큰이 없습니다."
                )

                maximum = max(maximum, length)

                if (index + 1) % 80 == 0:
                    print(
                        f"{split} 데이터 검사: "
                        f"{index + 1}/{len(rows)}개",
                        flush=True,
                    )

            print(
                f"{split} 데이터 검사 완료, 최대 길이 {maximum}토큰",
                flush=True,
            )

    return train_module

train_qwen.make_supervised_data_module = make_pilot_data

OriginalTrainer = train_qwen.Trainer

class PilotTrainer(OriginalTrainer):
    def train(self, *args, **kwargs):
        print("=== 학습 전 Internal Dev 평가 ===", flush=True)
        before = self.evaluate(metric_key_prefix="before")
        before_loss = float(before["before_loss"])
        assert math.isfinite(before_loss), \
            "학습 전 개발 손실이 정상적인 수치가 아닙니다."

        result = super().train(*args, **kwargs)

        print("=== 학습 후 Internal Dev 평가 ===", flush=True)
        after = self.evaluate(metric_key_prefix="after")
        after_loss = float(after["after_loss"])
        assert math.isfinite(after_loss), \
            "학습 후 개발 손실이 정상적인 수치가 아닙니다."

        report = {
            "학습률": self.args.learning_rate,
            "완료 단계": self.state.global_step,
            "학습 전 개발 손실": before_loss,
            "학습 후 개발 손실": after_loss,
            "개발 손실 감소량": before_loss - after_loss,
            "학습 결과": result.metrics,
        }

        destination = Path(self.args.output_dir) / "pilot_result.json"
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("=== 학습률 실험 결과 ===", flush=True)
        print(
            json.dumps(report, ensure_ascii=False, indent=2),
            flush=True,
        )
        return result

train_qwen.Trainer = PilotTrainer

try:
    train_qwen.train(attn_implementation="flash_attention_2")
finally:
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.destroy_process_group()
