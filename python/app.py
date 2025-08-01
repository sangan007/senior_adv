from flask import Flask, request, jsonify
import speech_recognition as sr
import google.generativeai as genai
import os
import json
import datetime

app = Flask(__name__)

GEMINI_API_KEY = "AIzaSyBE44Y5GcYy5MROKkZ0fzcKXM3zYuSwID4"
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')

@app.route('/process_voice_command', methods=['POST'])
def process_voice_command():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    audio_path = os.path.join('/tmp', audio_file.filename or 'temp_audio.wav')
    audio_file.save(audio_path)

    recognizer = sr.Recognizer()
    text_command = ""
    try:
        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            text_command = recognizer.recognize_google(audio_data, language='ml-IN')
            print(f"Recognized text (Malayalam): {text_command}")
    except sr.UnknownValueError:
        print("Speech Recognition could not understand audio")
        return jsonify({'action': 'unknown', 'message': 'Sorry, I could not understand your voice.'}), 200
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return jsonify({'action': 'unknown', 'message': 'Speech service unavailable. Please try again later.'}), 500
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    if not text_command:
        return jsonify({'action': 'unknown', 'message': 'No clear command detected.'}), 200

    try:
        prompt = f"""
        Analyze the following Malayalam voice command and extract the user's intent.
        Provide the output in JSON format.
        
        Expected JSON structure:
        {{
          "action": "set_reminder" | "query_reminder" | "delete_reminder" | "book_trip" | "get_tip" | "unknown",
          "time": "YYYY-MM-DDTHH:MM:SS" (ISO 8601 format, if applicable, for reminders),
          "message": "reminder message" (if applicable, for set_reminder),
          "trip_destination": "destination" (if applicable, for book_trip),
          "metadata": {{}} (any other relevant details)
        }}

        If a time is specified, convert it to an ISO 8601 string relative to the current time.
        Assume the current date is {datetime.date.today().isoformat()}.
        Consider common Malayalam phrases for time like "നാളെ" (tomorrow), "മറ്റന്നാൾ" (day after tomorrow),
        "രാവിലെ" (morning), "വൈകുന്നേരം" (evening), "ഇപ്പോൾ" (now), "അര മണിക്കൂർ കഴിഞ്ഞ്" (in half an hour).

        Voice Command: '{text_command}'
        """
        
        generation_config = {
            "response_mime_type": "application/json",
            "response_schema": {
                "type": "OBJECT",
                "properties": {
                    "action": {"type": "STRING"},
                    "time": {"type": "STRING", "nullable": True},
                    "message": {"type": "STRING", "nullable": True},
                    "trip_destination": {"type": "STRING", "nullable": True},
                    "metadata": {"type": "OBJECT", "nullable": True}
                },
                "required": ["action"]
            }
        }

        response = model.generate_content(
            [prompt],
            generation_config=generation_config
        )
        
        gemini_output = json.loads(response.text)
        print(f"Gemini Output: {gemini_output}")

        if gemini_output.get('action') == 'set_reminder' and 'time' in gemini_output and gemini_output['time']:
            try:
                parsed_time = datetime.datetime.fromisoformat(gemini_output['time'])
                if parsed_time.year == 1900:
                    parsed_time = parsed_time.replace(year=datetime.datetime.now().year)
                if parsed_time < datetime.datetime.now():
                    parsed_time = parsed_time + datetime.timedelta(days=1)
                gemini_output['time'] = parsed_time.isoformat()
            except ValueError:
                if "നാളെ" in text_command:
                    future_time = datetime.datetime.now() + datetime.timedelta(days=1)
                    gemini_output['time'] = future_time.replace(hour=8, minute=0, second=0, microsecond=0).isoformat()
                elif "മറ്റന്നാൾ" in text_command:
                    future_time = datetime.datetime.now() + datetime.timedelta(days=2)
                    gemini_output['time'] = future_time.replace(hour=8, minute=0, second=0, microsecond=0).isoformat()
                else:
                    gemini_output['time'] = None

        return jsonify(gemini_output), 200

    except Exception as e:
        print(f"Error with Gemini API: {e}")
        return jsonify({'action': 'unknown', 'message': 'Failed to process command with AI. Please try again.'}), 500

if __name__ == '__main__':
    if not os.path.exists('/tmp'):
        os.makedirs('/tmp')

    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)