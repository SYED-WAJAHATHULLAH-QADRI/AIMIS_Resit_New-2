import hashlib
import io
import json
import os
import tempfile
import time

import pandas as pd
import streamlit as st
from docx import Document
from google import genai
from google.genai import types
from jiwer import process_words
from pydantic import BaseModel, ConfigDict
import torch
import whisper


st.set_page_config(page_title="AIMIS", layout="wide")
st.title("AI Meeting Intelligence System")


# ---------- Output validation ----------

class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: str
    evidence: str


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: str
    owner: str
    deadline: str
    condition: str
    evidence: str


class MeetingOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    meeting_title: str
    summary: str
    decisions: list[Decision]
    action_items: list[Action]


def api_schema(value):
    if isinstance(value, dict):
        return {
            key: api_schema(item)
            for key, item in value.items()
            if key != "additionalProperties"
        }
    if isinstance(value, list):
        return [api_schema(item) for item in value]
    return value


PROMPTS = {
    "generic": (
        "Summarize the meeting and extract agreed decisions and "
        "future action items. Include supporting transcript quotes. "
        "Use empty strings for unspecified fields."
    ),
    "structured": """
Use only the supplied transcript.
Extract accepted choices or manager instructions as decisions.
Exclude suggestions without clear acceptance.
Extract future assignments and accepted commitments as actions.
Exclude completed work and general aspirations.
Separate independently executable tasks.
Keep a report and its required contents together.
Separate preparing and sharing a deliverable.
Record each distinct task once.
Use explicit owners; for ambiguous 'we', record the speaker
without inventing a team name. Otherwise leave the owner empty.
Preserve deadline wording and do not transfer deadlines between tasks.
Record conditional triggers separately.
Include exact supporting quotes.
Do not invent tasks, owners, dates or decisions.
"""
}


# ---------- Configuration ----------

def configured_key():
    key = os.getenv("GEMINI_API_KEY", "")
    try:
        key = st.secrets.get("GEMINI_API_KEY", key)
    except Exception:
        pass
    return key


with st.sidebar:
    st.header("Settings")
    entered_key = st.text_input("Gemini API key", type="password")
    api_key = entered_key or configured_key()
    model_name = st.text_input("Gemini model", "gemini-3.5-flash-lite")
    prompt_type = st.selectbox("Prompt", list(PROMPTS))
    st.caption("Use identical settings when comparing prompts.")


@st.cache_resource
def load_whisper():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model("small", device=device)


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------- Input ----------

uploaded = st.file_uploader(
    "Upload a meeting",
    type=["txt", "docx", "mp3", "wav", "m4a", "ogg", "flac", "mp4"]
)

if uploaded is None:
    st.info("Upload a transcript or meeting recording.")
    st.stop()

content = uploaded.getvalue()
input_hash = hashlib.sha256(content).hexdigest()
extension = os.path.splitext(uploaded.name)[1].lower()
is_audio = extension not in [".txt", ".docx"]

if st.session_state.get("input_hash") != input_hash:
    for key in list(st.session_state):
        if key.startswith("aimis_"):
            del st.session_state[key]
    st.session_state["input_hash"] = input_hash

if is_audio:
    st.audio(content)
    if st.button("Transcribe audio"):
        try:
            with st.spinner("Whisper is transcribing..."):
                temporary_path = None
                try:
                    with tempfile.NamedTemporaryFile(
                        suffix=extension, delete=False
                    ) as temporary:
                        temporary.write(content)
                        temporary_path = temporary.name

                    started = time.perf_counter()
                    result = load_whisper().transcribe(
                        temporary_path,
                        language="en",
                        temperature=0,
                        fp16=torch.cuda.is_available()
                    )
                    st.session_state["aimis_raw"] = result["text"].strip()
                    st.session_state["aimis_review_text"] = result["text"].strip()
                    st.session_state["aimis_asr_seconds"] = (
                        time.perf_counter() - started
                    )
                finally:
                    if temporary_path and os.path.exists(temporary_path):
                        os.unlink(temporary_path)
        except Exception as error:
            st.error(f"Transcription failed: {error}")
else:
    if "aimis_raw" not in st.session_state:
        try:
            if extension == ".docx":
                document = Document(io.BytesIO(content))
                text = "\n".join(
                    paragraph.text for paragraph in document.paragraphs
                    if paragraph.text.strip()
                )
            else:
                text = content.decode("utf-8-sig")

            st.session_state["aimis_raw"] = text
            st.session_state["aimis_review_text"] = text
        except Exception as error:
            st.error(f"Unable to read transcript: {error}")

if "aimis_raw" not in st.session_state:
    st.stop()


# ---------- Transcript review ----------

st.subheader("Review transcript")
reviewed_text = st.text_area(
    "Correct the text against the original input",
    key="aimis_review_text",
    height=300
)
review_confirmed = st.checkbox(
    "I checked this transcript against the original input.",
    key=f"aimis_confirm_{digest(reviewed_text)}"
)

st.download_button(
    "Download reviewed transcript",
    reviewed_text,
    file_name="reviewed_transcript.txt"
)

if is_audio and review_confirmed and reviewed_text.strip():
    import re

    def normalize(text):
        return " ".join(
            re.sub(r"[^\w\s]", " ", text.lower()).split()
        )

    reference = normalize(reviewed_text)
    raw = normalize(st.session_state["aimis_raw"])

    if reference:
        comparison = process_words(reference, raw)
        words = len(reference.split())
        st.write({
            "reference_words": words,
            "substitutions": comparison.substitutions,
            "deletions": comparison.deletions,
            "insertions": comparison.insertions,
            "WER_percent": round(100 * comparison.wer, 2)
        })
        st.caption(
            "WER = (substitutions + deletions + insertions) / "
            "reference words × 100. Reference: human-reviewed Whisper text."
        )


# ---------- Gemini extraction ----------

if st.button("Generate meeting output"):
    if not api_key:
        st.error("Enter your Gemini API key.")
    elif not reviewed_text.strip() or not review_confirmed:
        st.error("Check and confirm the transcript first.")
    else:
        try:
            with st.spinner("Generating output..."):
                started = time.perf_counter()
                response = genai.Client(api_key=api_key).models.generate_content(
                    model=model_name,
                    contents=(
                        PROMPTS[prompt_type]
                        + "\n\nMEETING TRANSCRIPT:\n"
                        + reviewed_text
                    ),
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        response_mime_type="application/json",
                        response_schema=api_schema(
                            MeetingOutput.model_json_schema()
                        )
                    )
                )
                validated = MeetingOutput.model_validate_json(
                    response.text or ""
                )
                st.session_state["aimis_output"] = validated.model_dump()
                st.session_state["aimis_run"] = {
                    "model": model_name,
                    "prompt": prompt_type,
                    "temperature": 0.2,
                    "transcript_sha256": digest(reviewed_text),
                    "processing_seconds": round(
                        time.perf_counter() - started, 3
                    )
                }
        except Exception as error:
            st.error(f"Generation failed: {error}")

if "aimis_output" not in st.session_state:
    st.stop()

if st.session_state["aimis_run"]["transcript_sha256"] != digest(reviewed_text):
    st.warning("Transcript changed. Generate a new output before evaluation.")
    st.stop()

output = st.session_state["aimis_output"]
run = st.session_state["aimis_run"]

st.subheader(output["meeting_title"])
st.write(output["summary"])
st.json(output)

st.download_button(
    "Download extraction and run details",
    json.dumps({"output": output, "run": run}, indent=2),
    file_name="meeting_output.json",
    mime="application/json"
)


# ---------- Human-reviewed evaluation ----------

st.subheader("Evaluate against reviewed annotations")
st.caption(
    "Upload a Cxx_reference_reviewed.json file from your project. "
    "Scores are computed from your explicit matches, not fuzzy similarity."
)

gold_file = st.file_uploader("Reference annotations", type=["json"])

if gold_file is None:
    st.stop()

try:
    gold = json.loads(gold_file.getvalue())
    if gold.get("status") != "human_reviewed":
        raise ValueError("Reference annotations must be human-reviewed.")

    for key in ["decisions", "action_items"]:
        if not isinstance(gold[key], list):
            raise ValueError(f"{key} must be a list.")

        ids = [item["id"] for item in gold[key]]
        if len(ids) != len(set(ids)):
            raise ValueError("Reference IDs must be unique.")

except Exception as error:
    st.error(f"Invalid reference: {error}")
    st.stop()

st.json(gold)

st.markdown("**Separate combined action tasks before matching.**")
st.caption(
    "Copy an action object to split it. Preserve its source_action, owner, "
    "deadline, condition and evidence. Mark repeated tasks as duplicates."
)

action_units = [
    {
        **item,
        "source_action": i,
        "exclude_duplicate": False,
        "duplicate_reason": ""
    }
    for i, item in enumerate(output["action_items"], 1)
]

evaluation_key = digest(json.dumps({
    "output": output, "run": run, "gold": gold
}, sort_keys=True))

units_text = st.text_area(
    "Action units JSON",
    value=json.dumps(action_units, indent=2),
    height=250,
    key=f"aimis_units_{evaluation_key}"
)

try:
    units = json.loads(units_text)
    if not isinstance(units, list):
        raise ValueError("Action units must be a list.")

    represented = set()
    for item in units:
        source = item["source_action"]
        if type(source) is not int or not 1 <= source <= len(action_units):
            raise ValueError("Invalid source_action.")

        represented.add(source)
        original = output["action_items"][source - 1]

        for field in ["owner", "deadline", "condition", "evidence"]:
            if item[field] != original[field]:
                raise ValueError(f"Preserve the generated {field}.")

        if not item["task"].strip():
            raise ValueError("Task cannot be empty.")
        if type(item["exclude_duplicate"]) is not bool:
            raise ValueError("exclude_duplicate must be true or false.")
        if item["exclude_duplicate"] and not item["duplicate_reason"].strip():
            raise ValueError("Explain duplicate exclusions.")

    if represented != set(range(1, len(action_units) + 1)):
        raise ValueError("Keep every original action represented.")

except Exception as error:
    st.error(f"Task units need correction: {error}")
    st.stop()

active_actions = [
    item for item in units if not item["exclude_duplicate"]
]

tables = {}
for key, prefix, predicted in [
    ("decisions", "D", output["decisions"]),
    ("action_items", "A", active_actions)
]:
    rows = []
    for i, item in enumerate(predicted, 1):
        rows.append({
            "output_id": f"{prefix}{i}",
            "task": item["task"],
            "reference_id": "",
            "judgment": "pending",
            "owner": "pending" if key == "action_items" else "N/A",
            "deadline": "pending" if key == "action_items" else "N/A",
            "condition": "pending" if key == "action_items" else "N/A",
            "reason": ""
        })

    st.markdown(f"**{key.replace('_', ' ').title()} matching**")
    if not rows:
        tables[key] = pd.DataFrame()
        st.write("No extracted items.")
        continue

    tables[key] = st.data_editor(
        pd.DataFrame(rows),
        disabled=["output_id", "task"],
        hide_index=True,
        column_config={
            "reference_id": st.column_config.SelectboxColumn(
                options=[""] + [item["id"] for item in gold[key]]
            ),
            "judgment": st.column_config.SelectboxColumn(
                options=["pending", "full", "partial", "unmatched"]
            ),
            **{
                field: st.column_config.SelectboxColumn(
                    options=["pending", "correct", "incorrect", "excluded", "N/A"]
                )
                for field in ["owner", "deadline", "condition"]
            }
        },
        key=f"aimis_matches_{key}_{evaluation_key}_{digest(units_text)}"
    )

st.caption(
    "Full means complete task meaning. Score fields only for full action "
    "matches. Exclude ambiguous reference owners or unassessable timing; "
    "explain exclusions. Unmatched does not automatically mean hallucinated."
)

reviewer_name = st.text_input("Evaluation reviewer")
evaluation_confirmed = st.checkbox(
    "I checked task units, matches and field judgments.",
    key=f"aimis_eval_confirm_{evaluation_key}_{digest(units_text)}"
)

if st.button("Calculate evaluation"):
    try:
        if not reviewer_name.strip() or not evaluation_confirmed:
            raise ValueError("Enter reviewer and confirm the evaluation.")

        task_scores = []
        field_scores = []

        for key in ["decisions", "action_items"]:
            records = tables[key].to_dict("records")
            reference_ids = {item["id"] for item in gold[key]}
            assigned = []

            for item in records:
                judgment = item["judgment"]
                if judgment not in ["full", "partial", "unmatched"]:
                    raise ValueError("Review every task judgment.")
                if not item["reason"].strip():
                    raise ValueError("Give a reason for every judgment.")

                if judgment == "unmatched":
                    if item["reference_id"]:
                        raise ValueError("Unmatched items need an empty reference ID.")
                else:
                    if item["reference_id"] not in reference_ids:
                        raise ValueError("Choose a valid reference ID.")
                    assigned.append(item["reference_id"])

                if key == "action_items" and judgment == "full":
                    reference_item = next(
                        target for target in gold[key]
                        if target["id"] == item["reference_id"]
                    )
                    for field in ["owner", "deadline", "condition"]:
                        if item[field] not in ["correct", "incorrect", "excluded"]:
                            raise ValueError("Review all fields of full action matches.")
                        if (
                            field == "owner"
                            and reference_item.get("owner_ambiguous", False)
                            and item[field] != "excluded"
                        ):
                            raise ValueError("Exclude ambiguous reference owners.")

            if len(assigned) != len(set(assigned)):
                raise ValueError("Reference matching must be one-to-one.")

            tp = sum(item["judgment"] == "full" for item in records)
            fp = len(records) - tp
            fn = len(gold[key]) - tp

            task_scores.append({
                "item_type": key,
                "TP": tp,
                "FP_not_fully_matched": fp,
                "FN": fn,
                "precision_percent": 100 * tp / (tp + fp) if tp + fp else None,
                "recall_percent": 100 * tp / (tp + fn) if tp + fn else None,
                "f1_percent": 200 * tp / (2 * tp + fp + fn)
                if 2 * tp + fp + fn else None
            })

            if key == "action_items":
                full = [item for item in records if item["judgment"] == "full"]
                for field in ["owner", "deadline", "condition"]:
                    correct = sum(item[field] == "correct" for item in full)
                    incorrect = sum(item[field] == "incorrect" for item in full)
                    excluded = sum(item[field] == "excluded" for item in full)
                    assessed = correct + incorrect
                    field_scores.append({
                        "field": field,
                        "correct": correct,
                        "assessed": assessed,
                        "excluded": excluded,
                        "accuracy_percent": 100 * correct / assessed
                        if assessed else None
                    })

        st.dataframe(pd.DataFrame(task_scores), hide_index=True)
        st.dataframe(pd.DataFrame(field_scores), hide_index=True)

        st.write(
            "Precision = TP/(TP+FP); Recall = TP/(TP+FN); "
            "F1 = 2TP/(2TP+FP+FN). Multiply by 100 for percentages."
        )
        st.write("Field accuracy = correct/assessed × 100.")

        report = {
            "reviewer": reviewer_name,
            "run": run,
            "reference": gold,
            "action_units": units,
            "matches": {
                key: table.to_dict("records")
                for key, table in tables.items()
            },
            "task_metrics": task_scores,
            "field_metrics": field_scores,
            "limitations": [
                "Extraction evaluated from reviewed transcript.",
                "References may be AI-assisted and human-reviewed.",
                "Field accuracy is conditional on fully matched actions.",
                "Unmatched items are not automatically hallucinations."
            ]
        }

        st.download_button(
            "Download evaluation evidence",
            json.dumps(report, indent=2, allow_nan=False),
            file_name="evaluation_report.json",
            mime="application/json"
        )

    except Exception as error:
        st.error(f"Evaluation not completed: {error}")
