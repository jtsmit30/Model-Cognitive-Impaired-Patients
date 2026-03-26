# Model-Cognitive-Impaired-Patients
Evaluating the markers within an annotated conversation transcript using classifiers and generating a report on an edge AI system.

The goal for this project is to create a system for evaluating wellbeing of a patient with a cognitive imapairment.

The model is split into 3 different layers.
Layer 1:
Sensors such as audio is analysed by open source models that produces these paralinguistic features of each utterence:
tone (flat, warm, anxious, angry, sad, ...) emotion2vec / SenseVoice
energy (very_low, low, medium, high) openSMILE (RMS energy)
pace (slow, normal, fast)  Whisper timestamps / syllable rate
timestamps (syllable rate)  pyannote diarization + Whisper timestamps
pause_before (seconds (float))  openSMILE / Parselmouth (Praat)
jitter% (float, typical 0.5–3.0)
pitch_mean Hz (float)
pitch_range Hz (float)

Layer 2:
Layer 2 runs the trained classifier models on the same utterances. This layer handles the 25 markers that require semantic understanding.

The classifier architecture uses a shared encoder with multiple classification heads. A single DeBERTa-v3-large model encodes each utterance (with paralinguistic annotations appended as special tokens), and separate classification heads predict each marker group.

Affective Classifier, 8 sigmoid outputs (one per marker, 0–1 probability); severity softmax (4-class) per active marker
- [CLS] utterance_text [SEP] tone:X energy:X pace:X [SEP]
Cognitive Semantic Classifier, 4 sigmoid outputs; severity per active marker. COG-04 (repetition) requires session-level window of past utterances as context
- [CLS] utterance_text [SEP] context_of_prior_2_utterances [SEP]
Behavioral Semantic Classifier, 4 sigmoid outputs; severity per active marker. These require the conversational exchange, not just the patient utterance
- [CLS] full_exchange (other_speaker + patient_response) [SEP] tone:X energy:X [SEP]
Self-Reference Classifier, 3 sigmoid outputs; severity per active marker. SLF-04 (perceived burden) outputs additional risk_flag boolean
- [CLS] utterance_text [SEP]
Somatic Content Classifier, 6 sigmoid outputs; severity per active marker. Also extracts mentioned symptom keywords for the clinical note
- [CLS] utterance_text [SEP]
Concordance Classifier, 4 sigmoid outputs. These specifically compare text sentiment against paralinguistic values 
- [CLS] utterance_text [SEP] tone:X energy:X pace:X jitter:X pitch:X [SEP]


After processing all utterances, Layer 2 aggregates utterance-level detections into session-level marker summaries. For each marker, the aggregation computes: detection frequency (how many utterances triggered this marker), maximum severity observed, severity trajectory across the session (getting worse, stable, improving), and the most salient evidence utterances (top 3 by detection confidence). This aggregation produces the structured marker report that Layer 3 consumes.

Layer 3:
The fine-tuned LLM receives a structured input combining all outputs from Layers 1 and 2. This is a much simpler task for the LLM than the pure distillation approach, because it does not need to detect any markers itself — it only needs to reason about pre-detected markers and generate clinical documentation.
The LLM’s input prompt contains the session metadata (patient ID, date, setting, duration, participants), the complete Layer 1 output (deterministic feature values and severity classifications), the complete Layer 2 output (marker detections with confidence, severity, evidence utterances), the patient’s baseline profile (historical averages for all metrics), and explicit instructions to produce proxy clinical scale scores, a SOAP-format clinical note, alert determinations, and recommended actions.
Because the LLM is not doing marker detection, it can focus entirely on clinical reasoning: weighing the relative importance of different marker combinations, identifying patterns that span multiple domains (e.g., combined AFF-03 + SOM-01 + BEH-03 suggesting a depressive episode), and generating clear, actionable clinical language.





Future Features:
Features to be added are computing features from sensor information algorithmically:
COG-01: Lexical Diversity
COG-03: Syntactic Simplification
COG-07: Pronoun Ambiguity
COG-08: Information Density
BEH-01: Response Latency
BEH-02: Response Brevity

