from flask import Flask, request, jsonify, render_template, abort
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time, json
from jsonschema import validate as json_validate, ValidationError
from .config import Config
from .logging_utils import setup_logging
from .auth import require_basic_auth
from .registry import Registry
from .runtime import ModelRuntime

setup_logging()

app = Flask(__name__, template_folder="templates", static_folder="static")
registry = Registry()
runtime = ModelRuntime()

REQUEST_COUNT = Counter("mlserve_requests_total", "Total requests", ["endpoint", "model", "status"])
LATENCY = Histogram("mlserve_request_latency_seconds", "Request latency (s)", ["endpoint", "model"])

@app.get("/")
def index():
    models = registry.list_models()
    return render_template("index.html", models=models, cfg=Config)

@app.get("/train")
def train_page():
    return render_template("train.html")

@app.post("/train")
def train_csv():
    require_basic_auth()
    name = request.form.get("name")
    version = request.form.get("version") or "1.0.0"
    target = request.form.get("target")
    algo = request.form.get("algo") or "auto"
    csv_file = request.files.get("csv")
    if not all([name, target, csv_file]):
        abort(400, description="Missing required fields: name, target, csv")

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    import joblib, os, json, shutil

    df = pd.read_csv(csv_file)
    if target not in df.columns:
        abort(400, description=f"Target column '{target}' not found. Columns: {list(df.columns)}")
    y = df[target]
    X = df.drop(columns=[target])

    cat_cols = [c for c in X.columns if X[c].dtype == 'object']
    num_cols = [c for c in X.columns if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols),
        ]
    )

    # choose task
    if algo in ("logreg","rf_clf"):
        is_classification = True
    elif algo in ("linreg","rf_reg"):
        is_classification = False
    else:
        is_classification = (y.dtype == 'object') or (y.nunique() <= 20)

    if is_classification:
        model = RandomForestClassifier(n_estimators=200, random_state=42) if algo=="rf_clf" else LogisticRegression(max_iter=1000)
    else:
        model = RandomForestRegressor(n_estimators=200, random_state=42) if algo=="rf_reg" else LinearRegression()

    pipe = Pipeline(steps=[('pre', pre), ('model', model)])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(Xtr, ytr)
    score = pipe.score(Xte, yte)

    # Save artifact
    os.makedirs(os.path.join(Config.ARTIFACT_DIR, name, version), exist_ok=True)
    artifact_path = os.path.join(Config.ARTIFACT_DIR, name, version, "trained.joblib")
    joblib.dump(pipe, artifact_path)

    # schema: list of objects with original feature names
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"{name}_request",
        "type": "object",
        "properties": {
            "inputs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {col: {"type": "number"} for col in num_cols} | {col: {"type": "string"} for col in cat_cols},
                    "required": num_cols + cat_cols,
                    "additionalProperties": False
                }
            }
        },
        "required": ["inputs"],
        "additionalProperties": False
    }
    schema_json = json.dumps(schema)

    # wrap local file to fit registry.register interface
    import os, shutil  # make sure these are imported at the top of the file

    class LocalFile:
        def __init__(self, src, filename):
            self.filename = filename
            self._src = src
        def save(self, dst):
            # Skip copy if source and destination are the same (Windows-safe)
            if os.path.abspath(self._src) == os.path.abspath(dst):
                return
            shutil.copyfile(self._src, dst)


    fwrap = LocalFile(artifact_path, "trained.joblib")
    out = registry.register(name=name, version=version, framework="sklearn", file_storage=fwrap, input_schema=schema_json)

    return jsonify({"ok": True, "trained": {"name": name, "version": version, "artifact": artifact_path, "score": score}, "registered": out})

@app.post("/models/register")
def register_model():
    require_basic_auth()
    name = request.form.get("name")
    version = request.form.get("version")
    framework = request.form.get("framework")
    f = request.files.get("artifact")
    input_schema = request.files.get("input_schema")
    if not all([name, version, framework, f]):
        abort(400, description="Missing required fields: name, version, framework, artifact")
    schema_json = None
    if input_schema:
        schema_json = input_schema.read().decode("utf-8")
        try:
            json.loads(schema_json)
        except Exception:
            abort(400, description="input_schema must be valid JSON")
    out = registry.register(name=name, version=version, framework=framework, file_storage=f, input_schema=schema_json)
    return jsonify({"ok": True, "model": out})

@app.post("/models/<name>/activate")
def activate_version(name):
    require_basic_auth()
    payload = request.get_json(force=True, silent=True) or {}
    version = payload.get("version")
    if not version:
        abort(400, description="Provide 'version' in JSON body")
    out = registry.activate(name, version)
    return jsonify({"ok": True, "activation": out})

@app.get("/models")
def list_models():
    return jsonify({"ok": True, "models": registry.list_models()})

@app.post("/predict/<name>")
def predict(name):
    start = time.time()
    status = "200"
    try:
        require_basic_auth()
        payload = request.get_json(force=True, silent=True) or {}

        from sqlalchemy import select
        from .db import SessionLocal, Model, ModelVersion

        with SessionLocal() as s:
            model = s.execute(select(Model).where(Model.name == name)).scalar_one_or_none()
            if not model:
                abort(404, description=f"Model '{name}' not found")
            active = next((mv for mv in model.versions if mv.active), None)
            if not active:
                abort(409, description=f"No active version for model '{name}'")

            if active.input_schema:
                schema = json.loads(active.input_schema)
                try:
                    json_validate(instance=payload, schema=schema)
                except ValidationError as e:
                    abort(400, description=f"Schema validation failed: {e.message}")

            m = runtime.load(active.framework, active.path)
            result = runtime.predict(active.framework, m, payload)
        return jsonify({"ok": True, "model": name, "version": active.version, "result": result})
    except Exception as e:
        status = "500"
        if hasattr(e, "code"):
            status = str(e.code)
        raise
    finally:
        LATENCY.labels(endpoint="/predict", model=name).observe(time.time() - start)
        REQUEST_COUNT.labels(endpoint="/predict", model=name, status=status).inc()

@app.get("/metrics")
def metrics():
    return generate_latest(), 200, {"Content-Type": CONTENT_TYPE_LATEST}

def main():
    app.run(host=Config.SERVER_HOST, port=Config.SERVER_PORT, debug=Config.FLASK_ENV == "development")

if __name__ == "__main__":
    main()
