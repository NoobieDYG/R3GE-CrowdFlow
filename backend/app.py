from flask import Flask, request, jsonify,Response,render_template
import sys
sys.path.append('C:/Users/Affaan Jaweed/Desktop/crowd_control_hackazrd')
from vision_model.crowd_count import process_crowd_video
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

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
        zones_sorted = sorted(zones, key=lambda x: x[0], reverse=True)
    
        # Build output dictionary with zone names ("zone_1", "zone_2", etc.)
        result = {}
        for index, (count, occupancy) in enumerate(zones_sorted, start=1):
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

if __name__ == '__main__':
    # Running on port 8000 to match your endpoint in send_json_data
    app.run(host="0.0.0.0", port=8000, debug=True)
