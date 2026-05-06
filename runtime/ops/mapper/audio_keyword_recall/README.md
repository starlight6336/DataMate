# AudioKeywordRecall ASR关键词召回率评估算子

## 概述

AudioKeywordRecall 将 `audio_preprocessor` 新增的关键词召回率能力封装为 DataMate Mapper 算子。算子读取中英文关键词列表与 ASR `merged_text.txt`，分别计算中文、英文关键词召回率，将结构化结果写入 `sample["ext_params"]["audio_keyword_recall"]`，并生成 `keyword_recall.txt` 报告。

## 功能特性

- **中英文评估**：中文使用关键词子串匹配，英文使用空格 token 匹配
- **Macro 平均**：按含关键词样本逐句计算召回率后取平均
- **报告输出**：生成包含逐句 hit/miss 明细的 `keyword_recall.txt`
- **可编排**：可单独接在 ASR 算子后，也可由端到端 ASR 流水线可选调用

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| zhKeywordPath | input | (空) | 中文关键词文件；留空使用 `audio_preprocessor/input_data/valiadation/zh_keyword.txt` |
| enKeywordPath | input | (空) | 英文关键词文件；留空使用 `audio_preprocessor/input_data/valiadation/en_keyword.txt` |
| hypPath | input | (空) | ASR 结果文件；留空使用 `audio_preprocessor/output_data/asr/merged_text.txt` |
| reportDir | input | (空) | 报告输出目录；留空使用 `audio_preprocessor/output_data/validation` |
| keepDetails | switch | true | 是否将逐句明细写入 `ext_params`；报告文件始终包含明细 |

## 输入输出

- **输入**：
  - `sample["filePath"]` 可为任意音频路径，仅用于保持 DataMate 音频链路编排一致
  - `zhKeywordPath` / `enKeywordPath`：Kaldi 风格关键词文件，每行 `utt_id kw1 kw2 ...`
  - `hypPath`：Kaldi 风格 ASR 结果文件，每行 `utt_id text...`
- **输出**：
  - `sample["text"]`：中英文关键词召回率摘要
  - `sample["ext_params"]["audio_keyword_recall"]`：召回率、样本数、报告路径与可选逐句明细

## 依赖说明

- 仅依赖 DataMate 运行时与 `loguru`
- 不依赖 ASR/LID/降噪模型，可独立打包为 zip 算子包

## 版本历史

- **v1.0.0**：首次发布，支持中英文关键词召回率评估
