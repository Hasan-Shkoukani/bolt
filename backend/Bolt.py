import os
import dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import InferenceClient
from google import genai
from google.genai import types
from gradio_client import Client
import re


dotenv.load_dotenv()

app = Flask(__name__)
CORS(app)
print("Flask loaded")

client = genai.Client(api_key=os.getenv("GEMINI_KEY"))
print("Gemini loaded")


label_map = {
    "LABEL_0": "Course Registration",
    "LABEL_1": "Documents & Certificates",
    "LABEL_2": "General Inquiry",
    "LABEL_3": "Payment & Fees",
    "LABEL_4": "Scheduling & Attendance"
}

def classify_text(text):
    client = Client("hshkoukani/bolt-space")
    result = client.predict(
            text=text,
            api_name="/predict"
    )
    return result


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
        print("Gemini generation error:", e)
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
        print("Classifier result:", result)

        match = re.match(r"(LABEL_\d+) \((\d+\.\d+)%\)", result)
        if match:
            label = match.group(1)
            score = float(match.group(2)) / 100
        else:
            label = result
            score = None
        mapped_label = label_map.get(label, label)
        gemini_response = generate_response(data.get('subject', ''), data.get('body', ''), mapped_label)

        return jsonify({
            "result": [{"label": mapped_label, "score": score}],
            "output": gemini_response
        })

    except Exception as e:
        return jsonify({"Error": str(e) or "Unknown server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
