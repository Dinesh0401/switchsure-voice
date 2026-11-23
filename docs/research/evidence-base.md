# Evidence base

This document records the initial evidence behind the architecture. It is a design input, not a claim that the proposed system already outperforms a baseline.

## Code-switching and multilingual ASR

- The 2026 systematic review of end-to-end code-switching ASR identifies switch regions, data scarcity, and language-pair variation as persistent problems: [Code-Switching in End-to-End Automatic Speech Recognition](https://aclanthology.org/2026.lrec-1.768/).
- A unified ASR and token language-ID model using a concatenated tokenizer reported strong language identification while emitting language IDs with text tokens. This supports explicit token-level LID rather than relying only on utterance-level language: [Unified Model for Code-Switching Speech Recognition and Language Identification](https://aclanthology.org/2023.calcs-1.7/).
- Linguistically constrained synthetic code-switching text improved an English-Spanish ASR system, suggesting augmentation should respect switching structure: [Improving Code-switched ASR with Linguistic Information](https://aclanthology.org/2022.coling-1.627/).
- MUCS describes multilingual and code-switching ASR data for Indian languages and is a candidate benchmark source: [Multilingual and code-switching ASR challenges for low resource Indian languages](https://arxiv.org/abs/2104.00235).
- Tamil accessibility research documents the importance of evaluation on older and transgender speakers and naturalistic service settings: [LT-EDI Tamil ASR 2025](https://aclanthology.org/2025.ltedi-1.16/) and [CEN_Amrita@LT-EDI 2024](https://aclanthology.org/2024.ltedi-1.21/).

## Recognition backbones

- Whisper was trained on 680,000 hours of multilingual and multitask supervision and is a strong reproducible baseline, not proof of code-switch reliability: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356).
- Hugging Face's Whisper implementation can return token timestamps and generation outputs needed for research features: [Whisper documentation](https://huggingface.co/docs/transformers/main/model_doc/whisper).
- faster-whisper offers word timestamps, VAD integration, segment language probabilities, multilingual segment detection, and hotword hints for the practical baseline: [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
- SeamlessM4T supports multilingual ASR and is an appropriate comparison adapter for selected languages: [SeamlessM4T](https://arxiv.org/abs/2308.11596) and [reference implementation](https://github.com/facebookresearch/seamless_communication).

## Confidence and uncertainty

- Decoder probabilities are not automatically reliable confidence scores. A learned temperature estimator improved equal-error rate and AUC-ROC for encoder-decoder ASR confidence: [Confidence Measures in Encoder-Decoder Models for Speech Recognition](https://www.isca-archive.org/interspeech_2020/woodward20_interspeech.html).
- Context-aware word-level confidence estimation reduced calibration errors for common and rare words in CTC systems: [Word-level Confidence Estimation for CTC Models](https://www.isca-archive.org/interspeech_2023/naowarat23b_interspeech.html).
- N-best disagreement plus acoustic and confidence references can support error detection and correction: [Error Correction by Paying Attention to Both Acoustic and Confidence References](https://www.isca-archive.org/interspeech_2024/shu24_interspeech.html).
- Confidence-weighted ensembling reduced WER in a low-resource setting, supporting adapter-level candidate diversity as an experiment: [Ensembles of Hybrid and End-to-End Speech Recognition](https://aclanthology.org/2024.lrec-main.547/).

## Critical entities

- Ordinary WER does not adequately measure downstream named-entity harm; task-sensitive measures can correlate better with NER impact: [How to evaluate ASR output for named entity recognition?](https://www.isca-archive.org/interspeech_2015/jannet15_interspeech.html).
- Contextual biasing substantially reduced entity-tag WER in the reported setting, supporting optional user/domain lexicons with distractor tests: [Contextual Density Ratio for Language Model Biasing](https://www.isca-archive.org/interspeech_2021/andresferrer21_interspeech.pdf).
- GLiNER provides open-type entity recognition; the multilingual v2.1 model card declares an Apache-2.0 license. It remains a candidate that must be validated on Tamil-English speech transcripts: [GLiNER model card](https://huggingface.co/urchade/gliner_base) and [paper](https://aclanthology.org/2024.naacl-long.300/).

## Accessibility

- WCAG 2.2 is the conformance baseline: [Web Content Accessibility Guidelines 2.2](https://www.w3.org/TR/wcag/).
- Current WCAG 3 draft work explicitly discusses alternatives to speech-only interaction and real-time text. It informs exploration but is not a conformance target: [WCAG 3 draft](https://www.w3.org/TR/wcag3/).

## Derived design hypotheses

1. Uncertainty will be higher near true switch boundaries than matched non-boundary words after controlling for frequency, acoustics, and entity status.
2. A boundary-aware confidence estimator will improve area under the risk-coverage curve over raw decoder probability.
3. Candidate-lattice entity protection plus contextual biasing will improve critical-entity recall and entity error rate without a material general-WER regression.
4. A calibrated review policy will reduce high-severity accepted errors at a tolerable user interruption rate.

Each hypothesis is falsifiable through the evaluation contract.
