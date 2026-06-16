import json
import os
import joblib
import numpy as np
import re
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import pipeline
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend", "templates"))
JSON_PATH = os.path.abspath(os.path.join(BASE_DIR, "business_model_labels.json"))
NVOCC_KEYWORDS = [ "baf", "bill of lading", "booking note", "b/l", "caf", "carrier allocation", "co-load", "coload", "container yard", "cy/cy", "demurrage", "detention", "fcl", "feu", "free days", "freight collect", "freight prepaid", "hbl", "high cube", "lcl", "liner", "mbl", "ocean carrier", "ocean freight", "pod", "pol", "port of discharge", "port of loading", "port to port", "p2p", "sea freight", "sea waybill", "terminal handling", "teu", "thc", "vessel", "voyage", "20ft", "40ft", "20gp", "40gp", "20hc", "40hc" ]
WAREHOUSE_KEYWORDS = [ "ambient", "bonded", "bundling", "cargo handling", "cfs", "cold storage", "crating", "cross-dock", "cross docking", "devanning", "distribution center", "forklift", "fulfillment", "goods received note", "grn", "inbound", "inventory", "labeling", "loading dock", "pallet", "palletized", "pick and pack", "pick & pack", "racking", "repacking", "shrink wrap", "skus", "sku", "storage", "stripping", "stuffing", "tally sheet", "temperature controlled", "unstuffing", "warehouse", "wms", "export", "exw" ]
FF_SUPPRESSION_KEYWORDS = [ "storage", "warehouse", "wms", "inventory", "cold storage", "fulfillment", "racking", "bonded storage" ]
ENCODER = SentenceTransformer("all-mpnet-base-v2")
print("Initialize")
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() else torch.float32)
device = 0 if torch.cuda.is_available() else -1
MODERNBERT_CLASSIFIER = pipeline("zero-shot-classification", model="MoritzLaurer/ModernBERT-large-zeroshot-v2.0", device=device, torch_dtype=dtype, trust_remote_code=True, batch_size=1)
torch.set_num_threads(os.cpu_count())
app = Flask(__name__, template_folder=TEMPLATE_DIR)
CORS(app)
def train_context_aware_model():
    if not os.path.exists(JSON_PATH):
        print(f"Error: Training source file missing at {JSON_PATH}")
        return
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    X = [item["email_text"] for item in data]
    y_nvocc = np.array([item.get("nvocc", 0) for item in data])
    y_warehouse = np.array([item.get("warehouse", 0) for item in data])
    y_ff = np.array([item.get("ff", 0) for item in data])
    print("Encoding contextual matrices...")
    X_vec = ENCODER.encode(X, show_progress_bar=True, normalize_embeddings=True)
    model_nvocc = LogisticRegression(max_iter=1000, class_weight="balanced")
    model_warehouse = LogisticRegression(max_iter=1000, class_weight="balanced")
    model_ff = LogisticRegression(max_iter=1000, class_weight="balanced")
    model_nvocc.fit(X_vec, y_nvocc)
    model_warehouse.fit(X_vec, y_warehouse)
    model_ff.fit(X_vec, y_ff)
    joblib.dump({
        "nvocc_model": model_nvocc,
        "warehouse_model": model_warehouse,
        "ff_model": model_ff
    }, os.path.join(BASE_DIR, "context_classifier.pkl"))
    print("Weights optimized. Checkpoint context_classifier.pkl generated.")
NEGATION_PATTERN = re.compile(
    r"\b(no|not|never|cannot|can't|won't|isn't|aren't|doesn't|don't|without|nor|neither|prohibited|disallowed|restricted|unauthorized)\b"
)
def is_negated(clause, kw_pos):
    window = clause[max(0, kw_pos - 25):kw_pos]
    return bool(NEGATION_PATTERN.search(window))
def count_keyword_hits(keywords, text):
    positive = 0
    negative = 0
    sentences = re.split(r'[.!?\n;]', text)
    for sentence in sentences:
        for kw in keywords:
            idx = sentence.find(kw)
            if idx != -1:
                if is_negated(sentence, idx):
                    negative += 1
                else:
                    positive += 1
    return positive, negative
def extract_matched_keywords(keywords, text):
    sentences = re.split(r'[.!?\n;]', text)
    matched = []
    seen = set()
    for sentence in sentences:
        for kw in keywords:
            idx = sentence.find(kw)
            if idx != -1 and kw not in seen:
                negated = is_negated(sentence, idx)
                matched.append({
                    "keyword": kw,
                    "negated": negated,
                    "sentence": sentence.strip()
                })
                seen.add(kw)
    return matched
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/classifier')
def classifier():
    return render_template('businessmodelemailclassifier.html')
@app.route('/predict', methods=['POST'])
def predict():
    req_data = request.get_json() or {}
    email_text = req_data.get("email_text", "")
    if not email_text.strip():
        return jsonify({"predicted_labels": ["Others"]})
    text_lower = email_text.lower()
    nvocc_pos, nvocc_neg = count_keyword_hits(NVOCC_KEYWORDS, text_lower)
    warehouse_pos, warehouse_neg = count_keyword_hits(WAREHOUSE_KEYWORDS, text_lower)
    ff_pos, ff_neg = count_keyword_hits(FF_KEYWORDS, text_lower)
    word_count = len(text_lower.split())
    model_path = os.path.join(BASE_DIR, "context_classifier.pkl")
    ml_nvocc = 0
    ml_warehouse = 0
    ml_ff = 0
    if os.path.exists(model_path):
        try:
            artifacts = joblib.load(model_path)
            features = ENCODER.encode([email_text], normalize_embeddings=True)
            ml_nvocc = artifacts.get("nvocc_model").predict(features)[0] if artifacts.get("nvocc_model") else 0
            ml_warehouse = artifacts.get("warehouse_model").predict(features)[0] if artifacts.get("warehouse_model") else 0
            ml_ff = artifacts.get("ff_model").predict(features)[0] if artifacts.get("ff_model") else 0
        except Exception as e:
            print(f"ML MODEL ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            ml_nvocc, ml_warehouse, ml_ff = 0, 0, 0
    candidate_labels = ["NVOCC", "Warehouse", "Freight Forwarding", "Others"]
    if word_count < 15:
        modernbert_scores = {"NVOCC": 0, "Warehouse": 0, "Freight Forwarding": 0, "Others": 1}
    else:
        modernbert_res = MODERNBERT_CLASSIFIER(email_text, candidate_labels, multi_label=True)
        modernbert_scores = dict(zip(modernbert_res["labels"], modernbert_res["scores"]))
    detected = []
    threshold = 0.5
    nvocc_kw_hits = extract_matched_keywords(NVOCC_KEYWORDS, text_lower)
    warehouse_kw_hits = extract_matched_keywords(WAREHOUSE_KEYWORDS, text_lower)
    ff_kw_hits = extract_matched_keywords(FF_KEYWORDS, text_lower)
    nvocc_score = (ml_nvocc * 0.4) + (modernbert_scores["NVOCC"] * 0.6)
    if nvocc_score >= threshold or (nvocc_pos >= 2 and nvocc_neg < nvocc_pos):
        if nvocc_neg == 0 or nvocc_pos > nvocc_neg:
            detected.append("NVOCC")
    warehouse_score = (ml_warehouse * 0.4) + (modernbert_scores["Warehouse"] * 0.6)
    if warehouse_score >= threshold or (warehouse_pos >= 2 and warehouse_neg < warehouse_pos):
        if warehouse_neg == 0 or warehouse_pos > warehouse_neg:
            detected.append("Warehouse")
    ff_score = (ml_ff * 0.4) + (modernbert_scores["Freight Forwarding"] * 0.6)
    if ff_score >= threshold or (ff_pos >= 3 and ff_neg < ff_pos):
        if ff_neg == 0 or ff_pos > ff_neg:
            detected.append("Freight Forwarding")
    if not detected or (word_count < 15) or (modernbert_scores["Others"] > 0.7):
        detected = ["Others"]
    justification = {}
    if "NVOCC" in detected:
        justification["NVOCC"] = nvocc_kw_hits
    if "Warehouse" in detected:
        justification["Warehouse"] = warehouse_kw_hits
    if "Freight Forwarding" in detected:
        justification["Freight Forwarding"] = ff_kw_hits
    return jsonify({
        "predicted_labels": detected,
        "justification": justification,
        "keyword_hits": {
            "nvocc": {"positive": nvocc_pos, "negative": nvocc_neg},
            "warehouse": {"positive": warehouse_pos, "negative": warehouse_neg},
            "ff": {"positive": ff_pos, "negative": ff_neg}
        },
        "deep_learning_verification": {
            "modernbert_scores": modernbert_scores
        }
    })
if __name__ == '__main__':
    pkl_path = os.path.join(BASE_DIR, "context_classifier.pkl")
    if os.path.exists(pkl_path) and os.path.exists(JSON_PATH):
        json_mtime = os.path.getmtime(JSON_PATH)
        pkl_mtime = os.path.getmtime(pkl_path)
        if (json_mtime - pkl_mtime) > 0.1:
            print("Changes in JSON file. Re-training models...")
            try:
                os.remove(pkl_path)
            except Exception:
                pass
    if not os.path.exists(pkl_path):
        train_context_aware_model()
    app.run(debug=True, port=5000)