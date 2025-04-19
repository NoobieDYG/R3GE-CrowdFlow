from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/crowd_data', methods=['POST'])
def crowd_data():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data received"}), 400

    zone_counts = data.get("zone_counts")
    zone_occupancy = data.get("zone_occupancy")
    
    if zone_counts is None or zone_occupancy is None:
        return jsonify({"error": "Missing required keys: 'zone_counts' and/or 'zone_occupancy'"}), 400

    # Convert values to float, if needed
    try:
        zone_counts = [float(x) for x in zone_counts]
        zone_occupancy = [float(x) for x in zone_occupancy]
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
    
    return jsonify(result)

if __name__ == '__main__':
    # Running on port 8000 to match your endpoint in send_json_data
    app.run(host="0.0.0.0", port=8000, debug=True)
