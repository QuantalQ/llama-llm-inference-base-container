import os
import json
from flask import Flask, request, jsonify
from ctransformers import AutoModelForCausalLM

app = Flask(__name__)

# -------------------------------------------------------------------------
# If you want to automatically download from HF, specify the repo and file:
MODEL_REPO = "unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF"
MODEL_FILE = "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
MODEL_TYPE = "llama"  # ctransformers supports "llama", "gpt_neox", "starcoder", etc.

# Alternatively, if you'd rather mount a local volume or
# copy the .gguf model into /app/model, you can do:
#   MODEL_REPO = "/app/model"
#   MODEL_FILE = "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"
# so it doesn’t try to download from HF each time.

model = None

@app.before_first_request
def load_model():
    """
    Loads the GGUF model into RAM once, before handling the first request.
    """
    global model
    # If you have multiple gguf files, pick the one you want to load
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        model_file=MODEL_FILE,
        model_type=MODEL_TYPE,
        # cache_dir="/app/model"  # Where to cache the downloaded model
    )
    app.logger.info("Model loaded successfully!")

@app.route("/ping", methods=["GET"])
def ping():
    """
    Health check endpoint. SageMaker requires a 200 to say the container is alive.
    """
    return "OK", 200

@app.route("/invocations", methods=["POST"])
def invocations():
    """
    SageMaker Inference endpoint.
    Expects JSON with a "prompt" key (and optionally other generation params).
    """
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    # Optionally parse other generation parameters from data, e.g. "max_new_tokens", "temperature", etc.

    # ctransformers usage:
    #   The call returns generated text as a Python string
    #   Example: output = model(prompt, max_new_tokens=128, temperature=0.8)

    output = model(prompt,
                   max_new_tokens=data.get("max_new_tokens", 2048),
                   temperature=data.get("temperature", 0.8),
                   top_p=data.get("top_p", 0.95),
                   )

    # Return the generated text as JSON
    return jsonify({"generated_text": output})

if __name__ == "__main__":
    # When running locally (docker run -p 8080:8080 ...):
    #   open http://localhost:8080/ping to test
    #   or POST to http://localhost:8080/invocations with JSON
    app.run(host="0.0.0.0", port=8080)