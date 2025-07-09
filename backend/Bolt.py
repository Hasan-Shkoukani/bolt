import os
import dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.pipelines import pipeline
from google import genai
from google.genai import types


app = Flask(__name__)
print("Flask loaded")

dotenv.load_dotenv()
CORS(app)

client = genai.Client(api_key=os.getenv("GEMINI_KEY"))
print("Gemini loaded")

model_path = "hshkoukani/bolt"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
print("AI Model loaded")


label_map = {
    "LABEL_0": "Course Registration",
    "LABEL_1": "Documents & Certificates",
    "LABEL_2": "General Inquiry",
    "LABEL_3": "Payment & Fees",
    "LABEL_4": "Scheduling & Attendance"
}

def generate_response(subject, body, label):
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



@app.route('/analyze-label', methods=['POST']) 
def analyze_label():
    

    data = request.get_json()
    
    if not data or 'body' not in data:
        return jsonify({"Error": "No text provided"}), 400
    
    text = data.get('subject', '') + ' ' + data.get('body', '')
    
    try:
        try:
            result = classifier(text)
        except Exception as e:
            print(f"Error classifying text: {e}")
            return jsonify({"Error": str(e)}), 500
        
        result[0]['label'] = label_map.get(result[0]['label'], result[0]['label'])
        
        try:
            gemini = generate_response(data.get('subject', ''), data.get('body', ''), result[0]['label'])
        except Exception as e:
            print(f"Error generating response: {e}")
            return jsonify({"Error": str(e)}), 500
        else:
            return jsonify({
                "result": result,
                "output": gemini
            })
    
    except Exception as e:
        return jsonify({"Error": str(e)}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
