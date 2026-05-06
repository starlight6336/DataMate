# -- encoding: utf-8 --

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from loguru import logger

from datamate.core.base_op import Mapper


Detail = Tuple[str, float, int, int, List[str], List[str]]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[6]


def _audio_preprocessor_root() -> Path:
    return _repo_root() / "audio_preprocessor"


def _resolve_path(path_value: str, default_path: Path) -> Path:
    value = str(path_value or "").strip()
    if not value:
        return default_path
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (_repo_root() / p).resolve()
    return p


def _as_bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}


def _read_kw_kaldi(path: Path) -> Dict[str, List[str]]:
    data: Dict[str, List[str]] = {}
    if not path.exists():
        return data
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                key, rest = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)
                key = parts[0]
                rest = parts[1] if len(parts) > 1 else ""
            if key:
                data[key] = [w for w in rest.split() if w]
    return data


def _read_kv_text(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not path.exists():
        return data
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if parts:
                data[parts[0]] = parts[1] if len(parts) > 1 else ""
    return data


def _compute_keyword_recall(
    kw_map: Dict[str, List[str]],
    hyp_map: Dict[str, str],
    *,
    use_substring_match: bool,
) -> Tuple[float, int, int, List[Detail]]:
    keys = set(kw_map.keys()) & set(hyp_map.keys())
    if not keys:
        return 0.0, 0, 0, []

    recalls: List[float] = []
    details: List[Detail] = []
    num_total = 0
    for key in sorted(keys):
        ref_kws = [w for w in kw_map.get(key, []) if w]
        num_total += 1
        if not ref_kws:
            continue

        ref_set: Set[str] = {w.lower() for w in ref_kws}
        hyp_text = hyp_map.get(key, "")
        if use_substring_match:
            hyp_text_lower = hyp_text.lower()
            hit_words = [w for w in ref_set if w and w in hyp_text_lower]
            miss_words = [w for w in ref_set if w not in hyp_text_lower]
        else:
            hyp_set: Set[str] = {t.lower() for t in hyp_text.split() if t}
            hit_words = [w for w in ref_set if w in hyp_set]
            miss_words = [w for w in ref_set if w not in hyp_set]

        if not ref_set:
            continue
        hit = len(hit_words)
        recall = hit / float(len(ref_set))
        recalls.append(recall)
        details.append((key, recall, hit, len(ref_set), sorted(hit_words), sorted(miss_words)))

    if not recalls:
        return 0.0, 0, num_total, details
    return sum(recalls) / len(recalls), len(recalls), num_total, details


def _write_report(
    report_path: Path,
    zh_kw_path: Path,
    en_kw_path: Path,
    hyp_path: Path,
    zh_result: Tuple[float, int, int, List[Detail]],
    en_result: Tuple[float, int, int, List[Detail]],
) -> None:
    zh_recall, zh_used, zh_total, zh_detail = zh_result
    en_recall, en_used, en_total, en_detail = en_result
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        f.write("ASR 关键词召回率评估报告\n")
        f.write(f"中文关键词: {zh_kw_path}\n")
        f.write(f"英文关键词: {en_kw_path}\n")
        f.write(f"识别结果: {hyp_path}\n\n")
        f.write(
            f"中文：交集样本总数 = {zh_total}，含关键词样本数 = {zh_used}，"
            f"关键词召回率 = {zh_recall * 100:.2f}%\n"
        )
        f.write(
            f"英文：交集样本总数 = {en_total}，含关键词样本数 = {en_used}，"
            f"关键词召回率 = {en_recall * 100:.2f}%\n\n"
        )

        def dump(title: str, details: List[Detail]) -> None:
            f.write(f"==== {title} 逐句明细 ====\n")
            if not details:
                f.write("（无可评估样本）\n\n")
                return
            for utt_id, recall, hit, ref_size, hit_words, miss_words in details:
                f.write(f"utt_id: {utt_id}\n")
                f.write(f"  recall: {recall * 100:.2f}% (hit={hit}, ref_kw={ref_size})\n")
                f.write(f"  hit_kw: {' '.join(hit_words) if hit_words else 'None'}\n")
                f.write(f"  miss_kw: {' '.join(miss_words) if miss_words else 'None'}\n\n")

        dump("中文", zh_detail)
        dump("英文", en_detail)


class AudioKeywordRecall(Mapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ap_root = _audio_preprocessor_root()
        self.zh_kw_path = _resolve_path(
            str(kwargs.get("zhKeywordPath", "")),
            ap_root / "input_data" / "valiadation" / "zh_keyword.txt",
        )
        self.en_kw_path = _resolve_path(
            str(kwargs.get("enKeywordPath", "")),
            ap_root / "input_data" / "valiadation" / "en_keyword.txt",
        )
        self.hyp_path = _resolve_path(
            str(kwargs.get("hypPath", "")),
            ap_root / "output_data" / "asr" / "merged_text.txt",
        )
        self.report_dir = _resolve_path(
            str(kwargs.get("reportDir", "")),
            ap_root / "output_data" / "validation",
        )
        self.keep_details = _as_bool(kwargs.get("keepDetails", True))

    def execute(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        start = time.time()
        if not self.hyp_path.exists():
            raise FileNotFoundError(f"ASR 识别结果不存在: {self.hyp_path}")

        zh_kw = _read_kw_kaldi(self.zh_kw_path)
        en_kw = _read_kw_kaldi(self.en_kw_path)
        if not zh_kw and not en_kw:
            raise FileNotFoundError(f"未找到关键词文件: {self.zh_kw_path} / {self.en_kw_path}")

        hyp = _read_kv_text(self.hyp_path)
        zh_result = _compute_keyword_recall(zh_kw, hyp, use_substring_match=True)
        en_result = _compute_keyword_recall(en_kw, hyp, use_substring_match=False)

        report_path = self.report_dir / "keyword_recall.txt"
        _write_report(report_path, self.zh_kw_path, self.en_kw_path, self.hyp_path, zh_result, en_result)

        zh_recall, zh_used, zh_total, zh_detail = zh_result
        en_recall, en_used, en_total, en_detail = en_result
        recall_report: Dict[str, Any] = {
            "zh": {
                "recall": round(zh_recall, 6),
                "used_utterances": zh_used,
                "total_intersection_utterances": zh_total,
            },
            "en": {
                "recall": round(en_recall, 6),
                "used_utterances": en_used,
                "total_intersection_utterances": en_total,
            },
            "artifacts": {
                "zh_keyword": str(self.zh_kw_path),
                "en_keyword": str(self.en_kw_path),
                "hyp": str(self.hyp_path),
                "report": str(report_path),
            },
        }
        if self.keep_details:
            recall_report["details"] = {
                "zh": zh_detail,
                "en": en_detail,
            }

        ext = sample.get(self.ext_params_key, {})
        if not isinstance(ext, dict):
            ext = {"_raw": ext}
        ext["audio_keyword_recall"] = recall_report
        sample[self.ext_params_key] = ext
        sample[self.text_key] = (
            f"zh_keyword_recall={zh_recall * 100:.2f}%, "
            f"en_keyword_recall={en_recall * 100:.2f}%"
        )
        sample[self.data_key] = b""

        logger.info(
            f"fileName: {sample.get(self.filename_key)}, method: AudioKeywordRecall costs {time.time() - start:6f} s"
        )
        return sample
