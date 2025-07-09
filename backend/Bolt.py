import os
import dotenv
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.pipelines import pipeline
from google import genai
from google.genai import types

dotenv.load_dotenv()

app = Flask(__name__)
CORS(app)
print("Flask loaded")

client = genai.Client(api_key=os.getenv("GEMINI_KEY"))
print("Gemini loaded")

hf_token = os.getenv("HF_TOKEN")
hf_model = "hshkoukani/bolt"

# Load model and tokenizer locally
print("Loading local HF model...")
tokenizer = AutoTokenizer.from_pretrained(hf_model, token=hf_token)
model = AutoModelForSequenceClassification.from_pretrained(hf_model, token=hf_token)
clf_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
print("HF pipeline loaded")

label_map = {
    "LABEL_0": "Course Registration",
    "LABEL_1": "Documents & Certificates",
    "LABEL_2": "General Inquiry",
    "LABEL_3": "Payment & Fees",
    "LABEL_4": "Scheduling & Attendance"
}

def classify_text(text):
    try:
        result = clf_pipeline(text)
        return result
    except Exception as e:
        print("Hugging Face classification error:")
        traceback.print_exc()
        raise

def generate_response(subject, body, label):
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            config=types.GenerateContentConfig(
                system_instruction=(
                    "You are replying to emails that I receive.\n"
                    "You will be provided with the subject, body, and label of an incoming email.\n"
                    "\n"
                    "Instructions:\n"
                    "- You are the **recipient** of the original email. Write a reply accordingly.\n"
                    "- If sender and recipient names are provided, **flip their roles** in your reply.\n"
                    "- If either name is missing, **do not invent or use a placeholder like [Sender Name]**. Just leave the greeting out unless necessary.\n"
                    "- Strictly output only the body of the response. Do not include the subject, sender, recipient, greeting, or signature unless it's contextually appropriate within the reply body.\n"
                    "- Match your tone to the given label (e.g., Complaint, Request, etc.).\n"
                )
            ),
            contents=f"Subject: {subject}\nBody: {body}\nLabel: {label}"
        )
        return response.text
    except Exception as e:
        print("Gemini generation error:")
        traceback.print_exc()
        raise

@app.route('/analyze-label', methods=['POST'])
def analyze_label():
    try:
        data = request.get_json()
        print("Received data:", data)

        if not data or 'body' not in data:
            return jsonify({"Error": "No text provided"}), 400

        text = f"{data.get('subject', '')} {data.get('body', '')}"

        result = classify_text(text)

        if isinstance(result, list) and result and "label" in result[0]:
            label = result[0]['label']
            mapped_label = label_map.get(label, label)
        else:
            return jsonify({"Error": "Unexpected response from classifier"}), 500

        gemini_response = generate_response(data.get('subject', ''), data.get('body', ''), mapped_label)

        return jsonify({
            "result": [{"label": mapped_label, "score": result[0].get("score", None)}],
            "output": gemini_response
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"Error": str(e) or "Unknown server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
