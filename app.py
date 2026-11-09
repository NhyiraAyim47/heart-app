from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import joblib
import pandas as pd

app = FastAPI(title="Heart Disease Predictor")

bundle = joblib.load("heart_model.pkl")
pipe, columns = bundle["pipeline"], bundle["columns"]


class Patient(BaseModel):
    age: int = Field(ge=1, le=120)
    blood_pressure: int = Field(ge=40, le=260)
    cholesterol: int = Field(ge=50, le=600)
    bmi: float = Field(ge=10, le=70)
    physical_activity: int = Field(ge=0, le=4)
    stress_level: int = Field(ge=1, le=10)
    sex: str
    smoker: str
    diabetes: str
    family_history: str


@app.post("/predict")
def predict(p: Patient):
    row = {
        "age": p.age,
        "blood_pressure": p.blood_pressure,
        "cholesterol": p.cholesterol,
        "bmi": p.bmi,
        "physical_activity": p.physical_activity,
        "stress_level": p.stress_level,
        "sex_Male": int(p.sex == "Male"),
        "smoker_Yes": int(p.smoker == "Yes"),
        "diabetes_Yes": int(p.diabetes == "Yes"),
        "family_history_Yes": int(p.family_history == "Yes"),
    }
    df = pd.DataFrame([row])[columns]
    prob = float(pipe.predict_proba(df)[0][1])
    return {
        "prediction": int(prob >= 0.5),
        "probability_heart_disease": round(prob, 3),
    }


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Heart Disease Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #E8EFF2;
    --panel: #FFFFFF;
    --ink: #14283A;
    --muted: #5A6D7C;
    --line: #D3DDE4;
    --field: #F5F8FA;
    --teal: #0E7C80;
    --teal-dark: #0A5E62;
    --rose: #CC3A52;
    --radius: 14px;
  }
  * { box-sizing: border-box; }
  html { -webkit-text-size-adjust: 100%; }
  body {
    margin: 0;
    background: var(--bg);
    color: var(--ink);
    font-family: "Manrope", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    font-size: 16px;
    line-height: 1.5;
  }
  .wrap { max-width: 1040px; margin: 0 auto; padding: 40px 20px 56px; }

  header { margin-bottom: 28px; max-width: 620px; }
  h1 { font-size: clamp(28px, 4vw, 40px); line-height: 1.1; margin: 0 0 10px; font-weight: 800; letter-spacing: -0.02em; }
  header p { margin: 0; color: var(--muted); }

  .layout { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(0, 1fr); gap: 24px; align-items: start; }

  form, .result {
    background: var(--panel);
    border-radius: var(--radius);
    border: 1px solid var(--line);
  }
  form { padding: 8px 28px 28px; }

  .group + .group { border-top: 1px solid var(--line); }
  .group-title { font-weight: 700; font-size: 17px; margin: 0; padding: 24px 0 16px; }
  .group-inner { padding: 0 0 24px; }

  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .field { display: flex; flex-direction: column; gap: 6px; min-width: 0; }
  .group-inner > * + * { margin-top: 18px; }
  .grid2 > .field { margin-top: 0; }
  .field > label, .seg legend, .range-head label { font-size: 14px; font-weight: 600; color: var(--ink); padding: 0; }

  .input-box { position: relative; }
  input[type="number"] {
    width: 100%;
    font: inherit;
    font-weight: 600;
    color: var(--ink);
    background: var(--field);
    border: 1.5px solid var(--line);
    border-radius: 10px;
    padding: 11px 64px 11px 14px;
    -moz-appearance: textfield;
  }
  input[type="number"]::-webkit-outer-spin-button,
  input[type="number"]::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
  .unit { position: absolute; right: 14px; top: 50%; transform: translateY(-50%); font-size: 13px; color: var(--muted); pointer-events: none; }

  :focus-visible { outline: 3px solid rgba(14, 124, 128, 0.45); outline-offset: 2px; }
  input[type="number"]:focus-visible { border-color: var(--teal); outline-offset: 0; }

  .range-head { display: flex; justify-content: space-between; align-items: baseline; }
  .range-val { font-weight: 800; font-variant-numeric: tabular-nums; }
  input[type="range"] { width: 100%; accent-color: var(--teal); margin: 6px 0 0; }
  .range-ends { display: flex; justify-content: space-between; font-size: 12.5px; color: var(--muted); }

  fieldset.seg { border: 0; padding: 0; margin: 0; min-width: 0; }
  .seg-opts { display: grid; grid-template-columns: 1fr 1fr; gap: 0; margin-top: 6px; background: var(--field); border: 1.5px solid var(--line); border-radius: 10px; padding: 3px; }
  .seg-opts label { position: relative; }
  .seg-opts input { position: absolute; opacity: 0; inset: 0; cursor: pointer; margin: 0; }
  .seg-opts span { display: block; text-align: center; padding: 8px 10px; border-radius: 7px; font-weight: 600; font-size: 15px; color: var(--muted); }
  .seg-opts input:checked + span { background: var(--ink); color: #fff; }
  .seg-opts input:focus-visible + span { outline: 3px solid rgba(14, 124, 128, 0.45); outline-offset: 2px; }

  .submit { margin-top: 26px; display: flex; gap: 12px; align-items: center; }
  button.primary {
    font: inherit; font-weight: 700; font-size: 16px;
    background: var(--teal); color: #fff; border: 0; border-radius: 10px;
    padding: 13px 26px; cursor: pointer;
    transition: background 0.15s;
  }
  button.primary:hover { background: var(--teal-dark); }
  button.primary:disabled { background: #7FB2B4; cursor: progress; }
  button.ghost { font: inherit; font-weight: 600; background: none; border: 0; color: var(--muted); padding: 13px 8px; cursor: pointer; text-decoration: underline; text-underline-offset: 3px; }
  button.ghost:hover { color: var(--ink); }

  .result { padding: 28px; position: sticky; top: 20px; }
  .result h2 { margin: 0 0 4px; font-size: 17px; font-weight: 700; }
  .gauge { display: block; width: 100%; max-width: 340px; margin: 18px auto 4px; overflow: visible; }
  .gauge .track { stroke: var(--line); }
  .gauge .fill { stroke: var(--teal); transition: stroke-dashoffset 0.9s cubic-bezier(.2,.8,.2,1), stroke 0.3s; }
  .gauge .pct { font-family: inherit; font-size: 34px; font-weight: 800; fill: var(--ink); font-variant-numeric: tabular-nums; }
  .gauge .cap { font-family: inherit; font-size: 11.5px; fill: var(--muted); }
  .gauge .cut { stroke: var(--ink); stroke-width: 2.5; stroke-linecap: round; }

  .verdict { text-align: center; font-size: 22px; font-weight: 800; letter-spacing: -0.01em; margin: 6px 0 6px; }
  .verdict.pos { color: var(--rose); }
  .verdict.neg { color: var(--teal); }
  .note { text-align: center; color: var(--muted); font-size: 14.5px; margin: 0 auto; max-width: 34ch; }
  .error { color: var(--rose); font-weight: 600; text-align: center; }

  .howto { margin: 22px 0 0; padding-top: 18px; border-top: 1px solid var(--line); font-size: 13.5px; color: var(--muted); }

  footer { margin-top: 28px; max-width: 700px; font-size: 13.5px; color: var(--muted); }

  @media (max-width: 820px) {
    .layout { grid-template-columns: 1fr; }
    .result { position: static; }
  }
  @media (max-width: 460px) {
    .grid2 { grid-template-columns: 1fr; }
    form { padding: 4px 18px 22px; }
  }
  @media (prefers-reduced-motion: reduce) {
    .gauge .fill { transition: none; }
  }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Heart Disease Predictor</h1>
    <p>Enter a patient's details and the model will predict whether they are likely to have heart disease, along with how sure it is.</p>
  </header>

  <div class="layout">
    <form id="form" novalidate>
      <section class="group">
        <h3 class="group-title">Vitals</h3>
        <div class="group-inner">
          <div class="grid2">
            <div class="field">
              <label for="age">Age</label>
              <div class="input-box"><input id="age" name="age" type="number" inputmode="numeric" min="1" max="120" value="55" required><span class="unit">years</span></div>
            </div>
            <div class="field">
              <label for="bmi">BMI</label>
              <div class="input-box"><input id="bmi" name="bmi" type="number" inputmode="decimal" step="0.1" min="10" max="70" value="28.5" required><span class="unit">kg/m²</span></div>
            </div>
          </div>
          <div class="grid2">
            <div class="field">
              <label for="blood_pressure">Blood pressure</label>
              <div class="input-box"><input id="blood_pressure" name="blood_pressure" type="number" inputmode="numeric" min="40" max="260" value="130" required><span class="unit">mmHg</span></div>
            </div>
            <div class="field">
              <label for="cholesterol">Cholesterol</label>
              <div class="input-box"><input id="cholesterol" name="cholesterol" type="number" inputmode="numeric" min="50" max="600" value="220" required><span class="unit">mg/dL</span></div>
            </div>
          </div>
        </div>
      </section>

      <section class="group">
        <h3 class="group-title">Lifestyle</h3>
        <div class="group-inner">
          <div class="field">
            <div class="range-head"><label for="physical_activity">Physical activity</label><span class="range-val" id="pa_val">2</span></div>
            <input id="physical_activity" name="physical_activity" type="range" min="0" max="4" step="1" value="2">
            <div class="range-ends"><span>0 (least active)</span><span>4 (most active)</span></div>
          </div>
          <div class="field">
            <div class="range-head"><label for="stress_level">Stress level</label><span class="range-val" id="st_val">6</span></div>
            <input id="stress_level" name="stress_level" type="range" min="1" max="10" step="1" value="6">
            <div class="range-ends"><span>1 (calm)</span><span>10 (very stressed)</span></div>
          </div>
          <div class="field">
            <fieldset class="seg">
              <legend>Smoker</legend>
              <div class="seg-opts">
                <label><input type="radio" name="smoker" value="Yes" checked><span>Yes</span></label>
                <label><input type="radio" name="smoker" value="No"><span>No</span></label>
              </div>
            </fieldset>
          </div>
        </div>
      </section>

      <section class="group">
        <h3 class="group-title">Background</h3>
        <div class="group-inner">
          <div class="field">
            <fieldset class="seg">
              <legend>Sex</legend>
              <div class="seg-opts">
                <label><input type="radio" name="sex" value="Male" checked><span>Male</span></label>
                <label><input type="radio" name="sex" value="Female"><span>Female</span></label>
              </div>
            </fieldset>
          </div>
          <div class="grid2">
            <div class="field">
              <fieldset class="seg">
                <legend>Diabetes</legend>
                <div class="seg-opts">
                  <label><input type="radio" name="diabetes" value="Yes"><span>Yes</span></label>
                  <label><input type="radio" name="diabetes" value="No" checked><span>No</span></label>
                </div>
              </fieldset>
            </div>
            <div class="field">
              <fieldset class="seg">
                <legend>Family history</legend>
                <div class="seg-opts">
                  <label><input type="radio" name="family_history" value="Yes" checked><span>Yes</span></label>
                  <label><input type="radio" name="family_history" value="No"><span>No</span></label>
                </div>
              </fieldset>
            </div>
          </div>
        </div>
      </section>

      <div class="submit">
        <button type="submit" class="primary" id="go">Check risk</button>
        <button type="button" class="ghost" id="reset-btn">Reset</button>
      </div>
    </form>

    <section class="result" aria-labelledby="res-title">
      <h2 id="res-title">Prediction</h2>
      <svg class="gauge" viewBox="0 0 200 124" role="img" aria-label="Probability gauge">
        <path class="track" d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke-width="14" stroke-linecap="round"/>
        <path class="fill" id="fill" d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke-width="14" stroke-linecap="round" stroke-dasharray="251.33" stroke-dashoffset="251.33"/>
        <line class="cut" x1="100" y1="6" x2="100" y2="34"/>
        <text class="pct" id="pct" x="100" y="92" text-anchor="middle">&ndash;</text>
        <text class="cap" x="100" y="112" text-anchor="middle">chance of heart disease</text>
      </svg>
      <div id="out" aria-live="polite">
        <p class="note" style="margin-top:18px">Fill in the details and select Check risk to see the model's prediction.</p>
      </div>
      <p class="howto">The dark marker at the top is the 50% cutoff. When the coloured arc passes it, the model predicts heart disease.</p>
    </section>
  </div>

  <footer>
    This is a learning project. The model was trained on a synthetic dataset, so its predictions are not reliable and must not be used for medical decisions.
  </footer>
</div>

<script>
  const $ = (id) => document.getElementById(id);
  const form = $("form"), fill = $("fill"), pct = $("pct"), out = $("out"), go = $("go");
  const ARC = 251.33;

  const defaults = {};
  new FormData(form).forEach((v, k) => defaults[k] = v);

  for (const [id, label] of [["physical_activity", "pa_val"], ["stress_level", "st_val"]]) {
    $(id).addEventListener("input", (e) => { $(label).textContent = e.target.value; });
  }

  function showEmpty() {
    fill.style.strokeDashoffset = ARC;
    fill.style.stroke = "";
    pct.textContent = "\u2013";
    out.innerHTML = '<p class="note" style="margin-top:18px">Fill in the details and select Check risk to see the model\u2019s prediction.</p>';
  }

  function showResult(r) {
    const p = r.probability_heart_disease;
    const positive = r.prediction === 1;
    const colour = positive ? "var(--rose)" : "var(--teal)";
    fill.style.stroke = colour;
    fill.style.strokeDashoffset = ARC * (1 - p);
    pct.textContent = Math.round(p * 100) + "%";
    const close = Math.abs(p - 0.5) < 0.1;
    const note = close
      ? "This is close to the 50% cutoff, so the model is not confident either way."
      : positive
        ? "The model leans towards heart disease for this profile."
        : "The model leans towards no heart disease for this profile.";
    out.innerHTML =
      '<div class="verdict ' + (positive ? "pos" : "neg") + '">' +
      (positive ? "Heart disease predicted" : "No heart disease predicted") +
      '</div><p class="note">' + note + '</p>';
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!form.checkValidity()) { form.reportValidity(); return; }
    const d = Object.fromEntries(new FormData(form));
    for (const k of ["age", "blood_pressure", "cholesterol", "physical_activity", "stress_level"]) d[k] = parseInt(d[k], 10);
    d.bmi = parseFloat(d.bmi);
    go.disabled = true; go.textContent = "Checking\u2026";
    try {
      const res = await fetch("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(d) });
      if (!res.ok) throw new Error("bad response");
      showResult(await res.json());
    } catch (err) {
      showEmpty();
      out.innerHTML = '<p class="error" style="margin-top:18px">The prediction could not be made. Check that every value is filled in and within range, then try again.</p>';
    } finally {
      go.disabled = false; go.textContent = "Check risk";
    }
  });

  $("reset-btn").addEventListener("click", () => {
    HTMLFormElement.prototype.reset.call(form);
    $("pa_val").textContent = defaults.physical_activity;
    $("st_val").textContent = defaults.stress_level;
    showEmpty();
  });
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE
