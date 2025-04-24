from flask import Flask, request, jsonify,Response,render_template,url_for
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from vision_model.crowd_count import process_crowd_video

from dotenv import load_dotenv
from openai import OpenAI
from flask_cors import CORS

FRONTEND_TEMPLATE_DIR = os.path.join(BASE_DIR, 'frontend', 'templates')
app = Flask(__name__, template_folder=FRONTEND_TEMPLATE_DIR)
CORS(app)
load_dotenv() 


client=OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY"),
)

app.config['pause_suggestion']=False
app.config['chat_history'] = []  #made a global variable to store chat history
app.config['latest_zone_data'] = None  #added a global variable to store latest zone data



@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/crowd_data', methods=['POST', 'GET'])
def crowd_data():
    if request.method == 'POST': #change here added if block to check for post request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        zone_counts = data.get("zone_counts")
        zone_occupancy = data.get("zone_occupancy")
    
        if zone_counts is None or zone_occupancy is None:
            return jsonify({"error": "Missing required keys: 'zone_counts' and/or 'zone_occupancy'"}), 400

        # Convert values to float, if needed
        try:
            zone_counts = [round(x) for x in zone_counts]
            zone_occupancy = [round((x),2) for x in zone_occupancy]
        except Exception as e:
            return jsonify({"error": "Invalid data format; please ensure numeric values are provided."}), 400

        # Combine zone counts and occupancy into tuples: (count, occupancy)
        zones = list(zip(zone_counts, zone_occupancy))
    
        # Sort the zones by count in descending order (highest count first)
        #zones_sorted = sorted(zones, key=lambda x: x[0], reverse=True)
    
        # Build output dictionary with zone names ("zone_1", "zone_2", etc.)
        result = {}
        for index, (count, occupancy) in enumerate(zones, start=1):
            result[f"zone_{index}"] = {"count": count, "occupancy": occupancy}


        app.config['latest_zone_data']=result #change here
        return jsonify(result)
    

    elif request.method == 'GET': #added elif block to handle get exception
        result = app.config.get('latest_zone_data')
        if result:
            return jsonify(result)
        else:
            return jsonify({"error": "No data available yet"}), 404
        

@app.route('/video_feed')
def video_feed():
    video_path = request.args.get('video_path')
    return Response(process_crowd_video(video_path),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_suggestion',methods=['GET'])
def llm_suggestion():
    if app.config.get('pause_suggestion'):
        return jsonify({"suggestion": ""})
    zone_data=app.config.get('latest_zone_data')
    if not zone_data:
        return jsonify({"error": "No zone data available"}), 404
    prompt_parts=[]
    for zone_name, values in zone_data.items():
        count=values['count']
        occupancy=values['occupancy']
        status=(
            "Safe" if occupancy<55 else
            "Moderate" if occupancy<85 else
            "Crowded - Take Action" if occupancy<99 else
            "Critical - Evacuate"
            )
        prompt_parts.append(f"{zone_name}: {count} people, {occupancy}% occupancy,Status: {status}")
    full_prompt="\n".join(prompt_parts) + "\nWhat safety and crowd control advice or actions do you suggest considering all the data?"
    response=client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role":"system","content":'''You are a safety and crowd control specializing in managing piligrimage crowd like umrah and hajj expert providing EXTREMELY accurate and precise advice which are useful for the user. "
            "IMPORTANT RULES:
            -Consider both counts and occupancy dont just give suggestions based on occupancy.
            -Use the real arabic names of the gates instead of saying "gate 1" or "gate 2".
            - Give suggestions more using the gates of the mosque and the zones of the mosque.
            - Consider ONLY the real gates of Masjid Al-Haram in Makkah, Saudi Arabia and NOT THE GATES OF PROPHET MUHAMMAD (PBUH) MOSQUE IN MADINA.
            - Provide suggestions based on the status of the zones and the number of people in each zone.
            -Provide information only if the status is "Crowded - Take Action" or "Critical - Evacuate" else just return the status itself.
            - Dont use clauses like 'Since the status' just return the status itself.
            - Provide only the most relevant and immediate actions to take.
            - Provide exactly 3 bullet points, each 1-2 sentences maximum
            - Focus only on immediate actionable steps for crowd management
            - Be direct and specific - no general advice or explanations
            - Prioritize the most crowded zones
            - Format as a simple bulleted list with no introductions or conclusions'''},

            {"role":"user","content":full_prompt}
        ],
        max_tokens=150, #changed tokens for more precise response
        temperature=0.7,
    )
    
    suggestion=response.choices[0].message.content
    app.config['chat_history'] = [
    {"role": "assistant", "content": suggestion}]  ##storing the chat history
    #print("Suggestion:", suggestion)  # Print the suggestion to the console for debugging
    return jsonify({"suggestion": suggestion})

@app.route('/chat_bot', methods=['POST'])
def chatbot():
    data = request.json
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "No message provided"}), 400
    
    
    app.config['pause_suggestion'] = True
    
    
    chat_history = app.config['chat_history'].copy()  
    
    
    chat_history.append({"role": "user", "content": user_message})
    
    
    zone_data = app.config.get('latest_zone_data', {})
    zone_summary = "\n".join(
        f"{zone}: {values['count']} people, {values['occupancy']}% occupancy"
        for zone, values in zone_data.items()
    )
    
    
    messages = [
        {"role": "system", "content": f"Zone data:\n{zone_summary}"},
        *chat_history  
    ]
    
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=150,
        temperature=0.7,
    )
    
    
    reply = response.choices[0].message.content
    app.config['chat_history'].append({"role": "user", "content": user_message})
    app.config['chat_history'].append({"role": "assistant", "content": reply})
    app.config['pause_suggestion'] = False
    return jsonify({"reply": reply})


@app.route('/text_to_speech',methods=['POST'])
def text_to_speech():
    data=request.get_json()
    text=data.get("text","")
    if not text:
        return jsonify({"error":"No text provided"}),400
    speech_file_path = os.path.join(app.root_path, "static", "speech.wav")
    os.makedirs(os.path.dirname(speech_file_path), exist_ok=True)
    model="playai-tts-arabic"
    voice="Nasser-PlayAI"
    response_format="wav"

    tts_response=client.audio.speech.create(
        model=model,
        voice=voice,
        input=text,
        response_format=response_format,
    )
    tts_response.write_to_file(speech_file_path)

    return jsonify({"success":True,"audio_url":url_for('static', filename='speech.wav')})



@app.route('/end_chat', methods=['POST'])
def end_chat():
    app.config['pause_suggestion']=False
    return jsonify({"message": "Chat ended and suggestion resumed."})

if __name__ == '__main__':
    #app.run(host="0.0.0.0", port=8000, debug=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port,debug=True)

