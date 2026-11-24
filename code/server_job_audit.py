from pathlib import Path
from datetime import datetime
import fcntl
import hashlib
import html
import json
import os
import re
import shutil
import traceback
import unicodedata

ROOT = Path("/shared_dir/mmdl")
BASE = ROOT / "data/source_audit/MMR1"
JOB = Path(__file__).resolve().parent
CURRENT = (
    ROOT
    / "data/engineering83_prepared_20261118"
    / "validation_20261118_230546_267103"
)
MFR = (
    ROOT
    / "data/external_candidates/mmfinereason_123k"
    / "cda6140bfba449bef9e57065a528b7424966e144"
)
REPO = "MMR1/MMR1-SFT"
REV = "0748ebf44e8e9b33dca28fc03af3836ad457e8b5"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(name, value):
    path = JOB / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def status(stage, **extra):
    save(
        "status.json",
        dict(
            stage=stage,
            updated=datetime.now().isoformat(),
            **extra,
        ),
    )
    print(stage, extra, flush=True)


def canonical(text):
    text = unicodedata.normalize("NFKC", html.unescape(text or ""))
    text = re.sub(
        r"<image[^>]*>|\[problem_image_\d+\]",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(r"^\s*Question\s*:\s*", "", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def array_items(stream, chunk=1024 * 1024):
    decoder = json.JSONDecoder()
    buffer, position, ended = "", 0, False

    def fill():
        nonlocal buffer, position, ended
        more = stream.read(chunk)
        buffer = buffer[position:] + more
        position = 0
        ended = not more

    def skip():
        nonlocal position
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer) or ended:
                return
            fill()

    skip()
    if buffer[position:position + 1] != "[":
        raise ValueError("Expected a JSON array")
    position += 1
    first = True

    while True:
        skip()
        if buffer[position:position + 1] == "]":
            position += 1
            skip()
            if buffer[position:] or stream.read(1):
                raise ValueError("Unexpected data after JSON array")
            return

        if not first:
            if buffer[position:position + 1] != ",":
                raise ValueError("Missing JSON array separator")
            position += 1
            skip()

        while True:
            try:
                value, stop = decoder.raw_decode(buffer, position)
                break
            except json.JSONDecodeError:
                if ended or len(buffer) - position > 16 * 1024 * 1024:
                    raise ValueError("Invalid or oversized JSON record")
                fill()

        position = stop
        first = False
        yield value


def main():
    os.environ["HF_HUB_OFFLINE"] = "0"

    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    import pyarrow.parquet as pq

    links = {
        item["id"]: item
        for item in read(CURRENT / "source_links.json")
    }
    items = [
        item
        for item in read(CURRENT.parent / "paired_candidates.json")
        if item["source"] == "MMR1"
    ]
    assert len(items) == 10, "Expected 10 MMR1 candidates"

    tables, lookup, matches, hints = {}, {}, {}, []

    for item in items:
        link = links[item["id"]]
        name = link["parquet_file"]

        if name not in tables:
            tables[name] = pq.read_table(
                MFR / name,
                columns=["id", "source", "question", "ori_question"],
            ).to_pylist()

        row = tables[name][link["resolved_zero_based_row"]]
        assert str(row["id"]) == str(item["source_id"])
        assert row["source"] == "MMR1"
        assert row["question"].strip() == item["original_question"].strip()

        matches[item["id"]] = []
        for field in ("question", "ori_question"):
            key = canonical(row.get(field))
            if key:
                lookup.setdefault(key, []).append((item["id"], field))

    status("checking_metadata")
    info = HfApi().dataset_info(
        REPO,
        revision=REV,
        files_metadata=True,
    )
    remote = next(
        item for item in info.siblings
        if item.rfilename == "sft.json"
    )
    assert remote.size == 9868577745, "Unexpected metadata file size"

    save(
        "source.json",
        dict(
            repository=REPO,
            revision=REV,
            filename="sft.json",
            size=remote.size,
            card_license="other",
        ),
    )

    arguments = dict(
        repo_id=REPO,
        repo_type="dataset",
        revision=REV,
        filename="sft.json",
        cache_dir=str(BASE / "hub_cache"),
    )

    try:
        local = Path(
            hf_hub_download(**arguments, local_files_only=True)
        )
    except LocalEntryNotFoundError:
        assert shutil.disk_usage(BASE).free >= 16 * 1024**3, (
            "At least 16 GiB free disk space is required"
        )
        status("downloading", bytes=remote.size)
        local = Path(hf_hub_download(**arguments))

    assert local.stat().st_size == remote.size, (
        "Downloaded file size mismatch"
    )

    status("hashing")
    digest = hashlib.sha256()
    with local.open("rb") as stream:
        for block in iter(
            lambda: stream.read(4 * 1024 * 1024),
            b"",
        ):
            digest.update(block)

    published = (
        remote.lfs.get("sha256")
        if isinstance(remote.lfs, dict)
        else getattr(remote.lfs, "sha256", None)
    )
    if published:
        assert digest.hexdigest() == published, (
            "Published SHA256 mismatch"
        )

    save(
        "file_verification.json",
        dict(
            local_file=str(local),
            sha256=digest.hexdigest(),
            published_sha256=published,
        ),
    )

    hint_positions = {
        int(item["source_id"]) + offset
        for item in items
        for offset in range(-3, 4)
    }
    count, complete = 0, False
    status("scanning", rows=0, matched=0)

    with local.open(encoding="utf-8-sig") as stream:
        for index, row in enumerate(array_items(stream)):
            assert isinstance(row, dict), "Expected object record"

            turns = row.get(
                "conversations",
                row.get("messages", []),
            )
            texts = [
                turn.get("value", turn.get("content", ""))
                for turn in turns
                if turn.get("from", turn.get("role"))
                in ("human", "user")
            ]
            texts = [text for text in texts if isinstance(text, str)]
            images = row.get("images", row.get("image", []))

            for text in texts:
                for candidate, field in lookup.get(
                    canonical(text), []
                ):
                    matches[candidate].append(
                        dict(
                            zero_based_row=index,
                            official_id=row.get("id"),
                            images=images,
                            matched_field=field,
                        )
                    )

            if index in hint_positions:
                hints.append(
                    dict(
                        zero_based_row=index,
                        questions=texts,
                        images=images,
                    )
                )

            count = index + 1
            if count % 10000 == 0:
                status(
                    "scanning",
                    rows=count,
                    matched=sum(bool(value) for value in matches.values()),
                )
                save("matches.json", matches)

            if all(matches.values()):
                break
        else:
            complete = True

    save("matches.json", matches)
    save("diagnostic_position_hints.json", hints)

    matched = sum(bool(value) for value in matches.values())
    result = dict(
        repository=REPO,
        revision=REV,
        scanned_rows=count,
        full_scan=complete,
        matched_candidates=matched,
        unmatched=[
            key for key, value in matches.items()
            if not value
        ],
        previous_train_question_links=69,
        total_train_question_links=69 + matched,
        scope=(
            "Train question matches only. Image identity and original "
            "source terms are not established. Position hints are "
            "not accepted as matches."
        ),
    )
    save("result.json", result)
    status("completed", **result)


if __name__ == "__main__":
    BASE.mkdir(parents=True, exist_ok=True)

    with (BASE / "audit.lock").open("w") as lock:
        try:
            fcntl.flock(
                lock,
                fcntl.LOCK_EX | fcntl.LOCK_NB,
            )
        except BlockingIOError:
            raise SystemExit("An MMR1 audit is already running")

        (BASE / "latest_job.txt").write_text(
            str(JOB),
            encoding="utf-8",
        )

        try:
            main()
        except Exception:
            save("error.json", dict(error=traceback.format_exc()))
            status(
                "failed",
                error_file=str(JOB / "error.json"),
            )
            raise
