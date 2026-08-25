import base64
import os
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from groq import Groq

load_dotenv()

# Inisialisasi Flask dengan menyesuaikan folder Frontend
app = Flask(__name__, template_folder="../Frontend", static_folder="../Frontend")

API_KEYS = [os.getenv("GROQ_API_KEY_1"), os.getenv("GROQ_API_KEY_2")]


def get_groq_response(contents, model="llama-3.2-11b-vision-preview"):
    for index, key in enumerate(API_KEYS):
        if not key:
            continue
        try:
            client = Groq(api_key=key)

            # Menambahkan instruksi agar AI mendukung semua bahasa secara otomatis
            system_message = {
                "role": "system",
                "content": "You are AI SMARTH GENERATIV, a helpful and ultra-fast multilingual AI assistant. You can understand and respond fluently in any language requested by the user.",
            }

            full_messages = [system_message] + contents

            completion = client.chat.completions.create(
                model=model,
                messages=full_messages,
                temperature=0.7,
                max_tokens=1024,
            )
            return completion.choices[0].message.content, None
        except Exception as e:
            error_msg = str(e).lower()
            if (
                "rate_limit" in error_msg
                or "429" in error_msg
                or "quota" in error_msg
            ):
                print(
                    f"API Key {index + 1} limit reached. Trying next key..."
                )
                continue
            else:
                return None, str(e)

    return None, "LIMIT_EXCEEDED"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        prompt = request.form.get("prompt", "")
        image_file = request.files.get("image")

        content_list = []

        if image_file:
            image_bytes = image_file.read()
            base64_image = base64.b64encode(image_bytes).decode("utf-8")
            mime_type = image_file.mimetype or "image/jpeg"

            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{base64_image}"},
            })

        if prompt:
            content_list.append({"type": "text", "text": prompt})

        if not content_list:
            return jsonify({"error": "Empty message"}), 400

        messages = [{"role": "user", "content": content_list}]
        response_text, error = get_groq_response(messages)

        if error == "LIMIT_EXCEEDED":
            return (
                jsonify(
                    {"error_type": "LIMIT_EXCEEDED", "message": "Limit is up"}
                ),
                429,
            )
        elif error:
            return jsonify({"error": error}), 500

        return jsonify({"response": response_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)

