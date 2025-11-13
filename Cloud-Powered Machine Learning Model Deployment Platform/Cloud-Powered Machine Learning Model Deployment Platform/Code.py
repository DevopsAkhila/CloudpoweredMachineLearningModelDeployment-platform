# This Python script outlines a detailed simulation of a model deployment system
# including upload, validation, activation, prediction API handling, and logging.

import time
import random
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ModelRegistry:
    def __init__(self):
        self.artifacts = {}
        self.active_version = None
        logging.info("ModelRegistry initialized.")

    def store_artifact(self, version: str, data: Any):
        self.artifacts[version] = {'data': data, 'state': 'uploaded', 'timestamp': time.time()}
        logging.info(f"Artifact for version {version} stored.")

    def write_registry_entry(self, version: str):
        if version in self.artifacts:
            self.artifacts[version]['registry_written'] = True
            logging.info(f"Registry entry written for version {version}.")
        else:
            logging.error("Attempted to write registry for missing artifact.")
            raise ValueError("Artifact must be uploaded before registry entry.")

    def validate(self, version: str) -> bool:
        logging.info(f"Validating version {version}...")
        time.sleep(1)  # Simulate processing time
        result = random.choice([True, True, False])  # Bias toward success
        if result:
            logging.info(f"Validation succeeded for version {version}.")
        else:
            logging.warning(f"Validation failed for version {version}.")
        return result

    def activate_version(self, version: str):
        if version not in self.artifacts:
            raise ValueError("Version does not exist.")
        if self.validate(version):
            self.active_version = version
            self.artifacts[version]['state'] = 'deployed'
            self.artifacts[version]['activated_at'] = time.time()
            logging.info(f"Version {version} activated and deployed.")
        else:
            logging.error(f"Validation failed for version {version}. Not activated.")

    def resolve_version(self) -> Optional[str]:
        return self.active_version

class PredictionAPI:
    def __init__(self, registry: ModelRegistry):
        self.registry = registry
        self.metrics = []
        logging.info("PredictionAPI initialized.")

    def validate_json_schema(self, payload: Dict[str, Any]) -> bool:
        valid = 'input' in payload and isinstance(payload['input'], list)
        logging.info(f"JSON Schema validation {'passed' if valid else 'failed'}.")
        return valid

    def run_inference(self, version: str, input_data: Any) -> Dict[str, Any]:
        start_time = time.time()
        # Simulated inference
        time.sleep(0.1)
        if random.random() < 0.95:
            predictions = [x * 2 for x in input_data]  # Dummy model logic
            latency = int((time.time() - start_time) * 1000)
            logging.info(f"Inference completed in {latency}ms.")
            return {'predictions': predictions, 'latency_ms': latency, 'version': version}
        else:
            logging.error("Inference engine failed.")
            raise RuntimeError("Inference engine failed")

    def emit_metrics(self, info: Dict[str, Any]):
        self.metrics.append(info)
        logging.info(f"Metrics recorded: {info}")

    def handle_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate_json_schema(payload):
            error_info = {"error": "Invalid schema", "code": 400, "context": payload}
            logging.warning(f"Bad Request: {error_info}")
            return error_info

        version = self.registry.resolve_version()
        if version is None:
            logging.error("No active version available.")
            return {"error": "No active version", "code": 503}

        try:
            result = self.run_inference(version, payload['input'])
            result['status'] = 200
            self.emit_metrics({"version": version, "latency": result['latency_ms']})
            return result
        except Exception as e:
            error_info = {"error": str(e), "code": 500, "context": payload}
            logging.exception("Inference error occurred.")
            return error_info

# Simulation
if __name__ == "__main__":
    registry = ModelRegistry()
    api = PredictionAPI(registry)

    # Deployment sequence
    registry.store_artifact('v1.0', {'model': 'dummy-model'})
    registry.write_registry_entry('v1.0')
    registry.activate_version('v1.0')

    # Prediction loop simulation
    test_payloads = [
        {'input': [1, 2, 3]},
        {'input': [5, 10]},
        {},  # Invalid
        {'input': 'not-a-list'}  # Invalid
    ]

    for payload in test_payloads:
        response = api.handle_request(payload)
        print("Response:", response)
        time.sleep(0.5)  # Simulate request interval

    print("All metrics:", api.metrics)