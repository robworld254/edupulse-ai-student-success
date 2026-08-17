from __future__ import annotations

from datetime import UTC, datetime
from html import escape


def assessment_html(
    *,
    registration_number: str,
    student_name: str,
    school: str,
    risk_band: str,
    dropout_probability: float,
    predicted_outcome: str,
    probabilities: dict[str, float],
    inputs: dict[str, object],
    actions: list[str],
    signals: list[str],
    model_version: str,
) -> str:
    reg = escape(registration_number or "Not supplied")
    name = escape(student_name or "Not supplied")
    rows = "".join(f"<tr><td>{escape(str(k))}</td><td>{escape(str(v))}</td></tr>" for k, v in inputs.items())
    probs = "".join(f"<li><strong>{escape(k)}</strong>: {v:.1%}</li>" for k, v in probabilities.items())
    action_items = "".join(f"<li>{escape(a)}</li>" for a in actions)
    signal_items = "".join(f"<li>{escape(signal)}</li>" for signal in signals)
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>EduPulse AI Assessment</title>
<style>
body{{font-family:Arial,sans-serif;margin:40px;color:#1f2937}} h1{{color:#8f1d2c;margin-bottom:4px}}
.badge{{display:inline-block;background:#f4ead5;color:#7b4f00;padding:6px 10px;border-radius:999px;font-weight:700}}
.score{{font-size:42px;font-weight:800;color:#8f1d2c}} table{{border-collapse:collapse;width:100%;margin-top:16px}}
td{{border-bottom:1px solid #e5e7eb;padding:9px 4px}} td:first-child{{font-weight:700;width:46%}}
.note{{margin-top:24px;padding:14px;background:#f8f7f4;border-left:4px solid #d6a84d;font-size:13px}}
</style></head><body>
<div style='font-size:13px;font-weight:700;letter-spacing:.08em'>EDUPULSE AI · ACADEMIC PROJECT PROTOTYPE</div>
<h1>EduPulse AI — Student Success Assessment</h1>
<p>Student name: <strong>{name}</strong><br>Registration number: <strong>{reg}</strong><br>School: <strong>{escape(school)}</strong><br>Generated: {datetime.now(UTC).strftime("%d %b %Y, %H:%M UTC")}</p>
<span class='badge'>{escape(risk_band)}</span><div class='score'>{dropout_probability:.0%}</div>
<p>Model-estimated probability of the Dropout class. Highest-probability outcome: <strong>{escape(predicted_outcome)}</strong>.</p>
<h2>Outcome probabilities</h2><ul>{probs}</ul>
<h2>Assessment inputs</h2><table>{rows}</table>
<h2>Model signals</h2><ul>{signal_items}</ul>
<h2>Suggested support pathway</h2><ul>{action_items}</ul>
<div class='note'>Research prototype only. The model is trained on the public UCI student-success dataset, not Kabarak University student records. It must not be used as the sole basis for grading, progression, funding, discipline or exclusion decisions. Model version {escape(model_version)}.</div>
</body></html>"""
