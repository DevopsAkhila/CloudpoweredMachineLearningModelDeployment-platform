import joblib

def _try_import(module_name):
    try:
        return __import__(module_name)
    except Exception:
        return None

torch = _try_import("torch")
onnxruntime = _try_import("onnxruntime")

class ModelRuntime:
    def __init__(self):
        self.cache = {}

    def load(self, framework: str, path: str):
        key = (framework, path)
        if key in self.cache:
            return self.cache[key]
        if framework == "sklearn":
            model = joblib.load(path)
        elif framework == "torch":
            if torch is None:
                raise RuntimeError("PyTorch not installed.")
            model = torch.jit.load(path) if path.endswith('.pt') else torch.load(path, map_location="cpu")
            model.eval()
        elif framework == "onnx":
            if onnxruntime is None:
                raise RuntimeError("onnxruntime not installed.")
            model = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider'])
        else:
            raise ValueError(f"Unsupported framework: {framework}")
        self.cache[key] = model
        return model

    def predict(self, framework: str, model, payload: dict):
        inputs = payload.get("inputs")
        if inputs is None:
            raise ValueError("Payload must include 'inputs'.")
        if framework == "sklearn":
            import numpy as np
            # Allow list of dicts (from auto-train) OR list of lists (manual schema)
            if isinstance(inputs, list) and inputs and isinstance(inputs[0], dict):
                # Convert list-of-dicts to DataFrame then back to array via columns order
                import pandas as pd
                df = pd.DataFrame(inputs)
                preds = model.predict(df).tolist()
            else:
                X = np.array(inputs)
                preds = model.predict(X).tolist()
            return {"predictions": preds}
        elif framework == "torch":
            import torch as T
            X = T.tensor(inputs, dtype=T.float32)
            with T.inference_mode():
                out = model(X)
            out = out.detach().cpu().numpy().tolist()
            return {"predictions": out}
        elif framework == "onnx":
            import numpy as np
            if isinstance(inputs, dict):
                ort_inputs = {k: (np.array(v) if not hasattr(v, 'dtype') else v) for k, v in inputs.items()}
            else:
                ort_inputs = {"input": np.array(inputs)}
            out = model.run(None, ort_inputs)
            return {"predictions": [o.tolist() for o in out]}
        else:
            raise ValueError(f"Unsupported framework: {framework}")
