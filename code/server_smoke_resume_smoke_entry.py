import os
import sys
from pathlib import Path

repo = Path(os.environ["SMOKE_REPO"])
data = Path(os.environ["SMOKE_DATA"])
sys.path.insert(0, str(repo))
sys.path.insert(0, str(repo / "qwenvl/train"))

import peft
import qwenvl.data as registry
from transformers import TrainerCallback
import train_qwen

registry.data_dict["local_smoke16"] = {
    "annotation_path": str(data / "train.json"),
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
    total = sum(parameter.numel() for parameter in model.parameters())

    print(f"학습 파라미터: {trainable:,}개", flush=True)
    print(f"전체 파라미터: {total:,}개", flush=True)
    print(f"학습 비율: {100 * trainable / total:.4f}%", flush=True)
    print("언어 모델 LoRA만 학습하도록 설정됐습니다.", flush=True)
    print("시각 인코더와 병합 모듈 동결을 확인했습니다.", flush=True)
    return model

peft.get_peft_model = checked_get_peft_model

class SmokeCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        expected = int(os.environ["EXPECTED_START"])
        assert state.global_step == expected, (
            f"시작 단계 불일치: 실제 {state.global_step}, 예상 {expected}"
        )

        optimizer = kwargs.get("optimizer")
        scheduler = kwargs.get("lr_scheduler")
        state_count = len(optimizer.state) if optimizer is not None else 0
        scheduler_step = getattr(scheduler, "last_epoch", None)

        print(f"학습 시작 단계: {state.global_step}", flush=True)
        print(f"옵티마이저 상태 항목 수: {state_count}", flush=True)
        print(f"학습률 스케줄러 단계: {scheduler_step}", flush=True)

        if expected:
            assert optimizer is not None and state_count > 0, \
                "옵티마이저 상태가 복원되지 않았습니다."
            assert scheduler_step == expected, \
                "학습률 스케줄러 단계가 예상과 다릅니다."

    def on_step_end(self, args, state, control, **kwargs):
        stop = int(os.environ.get("STOP_AFTER", "0"))
        if stop and state.global_step >= stop:
            control.should_save = True
            control.should_training_stop = True
        return control

OriginalTrainer = train_qwen.Trainer

class SmokeTrainer(OriginalTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_callback(SmokeCallback())

train_qwen.Trainer = SmokeTrainer
train_qwen.train(attn_implementation="flash_attention_2")
