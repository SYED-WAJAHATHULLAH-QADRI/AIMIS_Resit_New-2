# AI Meeting Intelligence Evaluation

Evaluates meeting text and audio using Whisper transcription,
human transcript review, Gemini extraction, and reviewed references.

## Dataset
16 inputs representing 13 distinct conversations.
Repeated inputs share conversation IDs and must not be treated
as independent meetings in aggregate analysis.

## Setup
Install Python dependencies:
`pip install -r requirements.txt`

Whisper also requires FFmpeg installed on the system.

Open AIMIS_Text_Audio_Evaluation.ipynb in Google Colab.
The notebook uses /content/AIMIS_Text_Audio_Evaluation.
Copy the contents of project/ to that location to reuse saved data.
Enter your Gemini API key through the notebook's hidden input.

## Evaluation
Task precision = TP / (TP + FP)
Task recall = TP / (TP + FN)
F1 = 2TP / (2TP + FP + FN)
Only reviewed full task matches count as TP.
Partial or unmatched items are not automatically hallucinations.

Field accuracy = correct / assessed fields.
Ambiguous owners and unassessable timing are excluded.
Field scores are conditional on matched actions.

WER = (substitutions + deletions + insertions) / reference words.

## Files
project/ contains inputs, transcripts, annotations, generated outputs,
review evidence, calculation evidence and result tables.

## Limitations
References are AI-assisted and reviewed by one human.
Extraction uses reviewed transcripts rather than uncorrected audio.
M01's corrected run is evaluated separately from its archived original.
Combined conversation-level summaries are not yet included.

## Notebook
Add the downloaded notebook to this repository.
Never commit an API key or credentials.
