# Model-Cognitive-Impaired-Patients

Evaluating wellbeing markers in annotated conversation transcripts using modular classifiers, deployed on an edge AI system.

## Overview

This project builds a three-layer pipeline that processes natural conversations between patients (primarily older adults) and their conversation partners to detect clinical markers and generate structured wellbeing reports.

**Layer 1 — Sensor Feature Extraction**
Open-source audio models (Whisper, pyannote, openSMILE, emotion2vec) produce paralinguistic features for each utterance: tone, energy, pace, timestamps, pause duration, jitter, pitch mean, and pitch range.

**Layer 2 — Classifier Inference**
A shared DeBERTa-v3-large encoder with six classification heads detects 25+ semantic markers across affective, cognitive, behavioral, self-reference, somatic, and concordance domains. Utterance-level detections are aggregated into session-level summaries.

**Layer 3 — Clinical Output**
A fine-tuned LLM receives structured marker data from Layers 1 and 2 and produces proxy clinical scale scores, SOAP-format notes, alert levels, and recommended actions.

## Project Structure

```
├── conversation_generation/    # Synthetic training data pipeline
│   ├── main.py                 # Entry point
│   ├── generate_conversations_testing.py
│   ├── PromptGenerationLogik.py
│   ├── JsonUtils.py
│   └── README.md               # Generation-specific docs
├── docs/                       # Architecture & training documentation
└── README.md                   # This file
```

## Current Status

The project is in the **data generation phase** — building the annotated conversation corpus that the Layer 2 classifiers will be trained on. See `conversation_generation/README.md` for details on the generation pipeline.

## Future Work

- Deterministic feature extractors for Layer 1 (lexical diversity, syntactic complexity, response latency, information density)
- Classifier training on generated corpus
- Layer 3 LLM fine-tuning
- Edge deployment optimization

## TODO for Computer Vision and Pose Detection

- Reset Waving deque if enough direction changes aren't made
- Try to normalize coordinates for sitting detection
  

  
