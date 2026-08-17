# EduPulse AI

**Student Success Early-Warning and Intervention Platform**

| Submission detail | Value |
|---|---|
| Student | **Stephen Robert Ochieng Nyambok** |
| Registration number | **CS/M/0113/01/24** |
| Unit | **COMP 413** |
| Lecturer | **Dr Andrew Kipkebut** |
| Institution | **Kabarak University** |
| Submission date | **17 August 2026** |

EduPulse AI is a supervised-learning academic prototype that estimates **Dropout**, **Enrolled**, and **Graduate** outcome probabilities from information available at the **end of Semester 1**. It is localized for a Kabarak University context, but it is not an official Kabarak system and was not trained or validated on Kabarak student records.

![EduPulse AI dashboard](deliverables/final%20UI%20screenshots/dashboard.png)

## What the portal does

- Presents a concise executive dashboard with real validation evidence.
- Runs individual assessments from seven lecturer-friendly inputs.
- Shows calibrated outcome probabilities, support priority, model signals, and supportive pathways.
- Provides three valid synthetic demo profiles and a downloadable assessment summary.
- Explores the public research cohort and validates CSV batch assessments.
- Documents feature-set experiments, five-model benchmarking, tuning, calibration, class metrics, confusion matrix, and permutation importance.

The system supports academic advising, Student Finance, Guidance & Counselling, and Dean of Students review. It must never be the sole basis for grading, funding, progression, discipline, or exclusion.

## Final model evidence

| Item | Result |
|---|---:|
| Selected algorithm | Support Vector Machine |
| Tuned training-CV Macro F1 | 0.665 |
| Final holdout Macro F1 | 0.630 |
| Final balanced accuracy | 0.628 |
| Final accuracy | 0.721 |
| Final weighted F1 | 0.709 |
| Dropout precision | 0.734 |
| Dropout recall | 0.729 |
| Dropout F1 | 0.731 |

These values come from the canonical `artifacts/results.json` produced by the real training run. The 885-record test partition was held out until final evaluation.

## Data and methodology

The model uses UCI Machine Learning Repository Dataset 697, *Predict Students' Dropout and Academic Success*: 4,424 Portuguese higher-education records, 36 original predictors, no missing cells, and no duplicate rows. DOI: `10.24432/C5MC89`; licence: CC BY 4.0.

```text
Immutable UCI source
  → stratified 80/20 split (seed 42)
  → training-only feature-set experiment
  → 5-fold CV benchmark of five classifiers
  → randomized tuning of the two strongest models
  → nested training-CV calibration selection
  → single final holdout evaluation
  → permutation importance + saved sklearn pipeline
  → Streamlit decision-support portal
```

The saved artifact accepts exactly seven fields: units registered, units passed, Semester 1 average mark, assessments completed, tuition-fee status, outstanding fee balance, and scholarship/sponsorship status. It internally engineers pass rate, units not passed, and assessments per unit. All Semester 2, identity, school/programme, demographic, parental, age, macroeconomic, and Portuguese course-code fields are excluded.

## Run locally on Windows

For first setup or a normal start, double-click `START_EDUPULSE.bat`. The launcher creates `.venv`, installs changed requirements, fetches UCI Dataset 697 if absent, trains only when artifacts are missing or outdated, and opens `http://localhost:8501`.

For a presentation after setup, double-click `RUN_EDUPULSE.bat`. Fast start uses the existing artifact and never retrains. The portal needs no internet after setup.

## Reproduce training and quality checks

```powershell
.\.venv\Scripts\python.exe -m scripts.train
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check app.py src scripts views tests
.\.venv\Scripts\python.exe -m compileall -q app.py src scripts views tests
```

Training regenerates the model, canonical JSON, metric tables, predictions, confusion matrix, feature importance, class distribution, calibration comparison, feature-set comparison, and metadata.

## Project structure

```text
app.py                    Streamlit entry point
src/                      Data, features, modelling, inference, explanation, UI
views/                    Four portal pages
scripts/                  Data, training, notebook and screenshot utilities
tests/                    Data, transformation, leakage, artifact, demo, batch tests
data/raw/                 Immutable UCI source copy
artifacts/                Model and machine-readable results
notebooks/                Executed final academic notebook
reports/figures/           Generated academic figures
deliverables/             Final report, presentation deck and interface evidence
```

## Submission documents

- [Final report — PDF](deliverables/EduPulse_AI_Final_Report.pdf)
- [Presentation deck — PDF](deliverables/EduPulse_AI_Presentation.pdf)

## Limitations and responsible use

Portuguese research data cannot establish validity for Kabarak or Kenyan students. The Enrolled class is difficult to distinguish, financial indicators may reflect structural disadvantage, probability estimates can be wrong, and performance can drift. Before operational use, complete institutional governance, a data-protection impact assessment, local retrospective and prospective validation, fairness and calibration review, role-based access, monitoring, student communication, and correction/appeal processes.

## Submission integrity

The repository includes the immutable source dataset, trained model, machine-readable evaluation artifacts, executed notebook, automated tests, application source, screenshots, final report and presentation deck. Reported metrics are read from `artifacts/results.json`; the portal and submission documents do not use separately typed performance values.

The system is an academic prototype and not an official Kabarak University deployment. Dataset provenance, geographic limitations and responsible-use constraints are documented throughout the repository.
