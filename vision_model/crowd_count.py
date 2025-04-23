
import cv2
import torch
import numpy as np
from torchvision import transforms
import requests
import json
import sys
sys.path.append('C:/Users/Affaan Jaweed/Desktop/crowd_control_hackazrd/vision_model')
from vision_script_cv import convert_url

#from vision_script_cv import convert_url

from csrnet_model import load_csrnet_model_1, load_csrnet_model_2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_1 = load_csrnet_model_1("vision_model\\weights\\csrnet_pretrained.pth", device)
model_2=load_csrnet_model_2("vision_model\\weights\\csrnet_pretrained_2.pth", device)
if device.type == "cuda":
    model_1 = model_1.half()  #check here
    model_2 = model_2.half()  #if not rtx comment this if block


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def density_to_heatmap(density_map):
    density_map = density_map.squeeze().cpu().numpy()
    density_map = cv2.GaussianBlur(density_map, (15, 15), 0)
    density_map /= (density_map.max() + 1e-6)
    heatmap = cv2.applyColorMap((density_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap[:, :, 0] = heatmap[:, :, 0] = (heatmap[:, :, 0] * 0.2).astype(np.uint8)
    heatmap[:, :, 1] = (heatmap[:, :, 1] * 0.5).astype(np.uint8)
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)
    heatmap_resized=cv2.resize(heatmap, (1280, 720))
    return heatmap_resized


def get_zone_occupancy_percentage(density_map, zones=(2, 2)):
    density_np = density_map.squeeze().cpu().numpy()
    h, w = density_np.shape
    total_density = density_np.sum()  
    zone_occupancy = []
    
    zh, zw = h // zones[0], w // zones[1]
    
    for i in range(zones[0]):
        for j in range(zones[1]):
            sub = density_np[i*zh:(i+1)*zh, j*zw:(j+1)*zw]
            zone_density = sub.sum()
            zone_percentage = (zone_density / (total_density + 1e-6)) * 100  
            zone_occupancy.append(round(zone_percentage, 2))
    
    return zone_occupancy


def get_zone_counts(density_map, zones=(2, 2)):
    
    density_np = density_map.squeeze().cpu().numpy()
    h, w = density_np.shape
    zone_counts = []
    
    
    zh, zw = h // zones[0], w // zones[1]
    
    for i in range(zones[0]):
        for j in range(zones[1]):
            
            sub = density_np[i*zh:(i+1)*zh, j*zw:(j+1)*zw]
            zone_count = sub.sum()  
            zone_counts.append(round(zone_count, 2))  
    
    return zone_counts


'''def send_json_data(zone_occupancy,zone_count, endpoint="http://localhost:8000/crowd_data"):
    try:
        payload = {"zone_counts": zone_count,
                   "zone_occupancy": zone_occupancy}
        headers = {"Content-Type": "application/json"}
        requests.post(endpoint, data=json.dumps(payload), headers=headers)
    except Exception as e:
        print("JSON send failed:", e)'''

def send_json_data(zone_occupancy, zone_count, endpoint="http://localhost:8000/crowd_data"):
    try:
        
        converted_count = [round(val.item()) if hasattr(val, 'item') else round(val) for val in zone_count]
        converted_occupancy = [round(float(val.item()),2) if hasattr(val, 'item') else round((val),2) for val in zone_occupancy]
        
        payload = {
            "zone_counts": converted_count,
            "zone_occupancy": converted_occupancy
        }
        
        headers = {"Content-Type": "application/json"}
        response = requests.post(endpoint, data=json.dumps(payload), headers=headers)
        
        
        print(f"Response status: {response.status_code}")
        if response.status_code != 200:
            print(f"Response content: {response.text}")
            
    except Exception as e:
        print("JSON send failed:", e)


'''def process_crowd_video(video_path):
    video_path=convert_url(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return
    frame_count=0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 10 != 0: #check here
            continue
        resized = cv2.resize(frame, (640, 480))
        input_tensor = transform(resized).unsqueeze(0).to(device)

        if device.type == "cuda":     ##check here
            input_tensor = input_tensor.half() ##if not rtx comment this if block

        with torch.no_grad():
            density_map = model_2(input_tensor)
            total_count = density_map.sum().item()

        
        heatmap = density_to_heatmap(density_map)
        overlay = cv2.addWeighted(resized, 0.5, heatmap, 0.5, 0)

        cv2.putText(overlay, f"Count: {int(total_count)}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        zone_count= get_zone_counts(density_map)
        zone_occupancy = get_zone_occupancy_percentage(density_map)
        send_json_data(zone_occupancy,zone_count)
        #print("Zone Occupancy Percentages:", zone_occupancy)
        #print("Zone Counts:", zone_count)
        
        cv2.imshow("Crowd Heatmap", overlay)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        cv2.waitKey(30)

    cap.release()
    cv2.destroyAllWindows()'''

def process_crowd_video(video_path):
    if video_path.startswith("https://www.youtube.com/watch?v="): #added this to check if the url is a youtube link or not otherwise directly use the path
        video_path = convert_url(video_path)
    elif video_path.startswith("https://youtu.be/"):
        video_path = convert_url(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 27 != 0:
            continue

        resized = cv2.resize(frame, (1280, 720))
        input_tensor = transform(resized).unsqueeze(0).to(device)

        if device.type == "cuda":
            input_tensor = input_tensor.half()  # Skip if not using RTX

        with torch.no_grad():
            density_map = model_2(input_tensor)
            total_count = density_map.sum().item()

        heatmap = density_to_heatmap(density_map)
        overlay = cv2.addWeighted(resized, 0.5, heatmap, 0.5, 0)
        h, w, _ = overlay.shape
        zones = [
            ((0, 0), (w // 2, h // 2)),          # Zone 1 - Top Left
            ((w // 2, 0), (w, h // 2)),          # Zone 2 - Top Right
            ((0, h // 2), (w // 2, h)),          # Zone 3 - Bottom Left
            ((w // 2, h // 2), (w, h))           # Zone 4 - Bottom Right
        ]
        zone_labels = ["Zone 1", "Zone 2", "Zone 3", "Zone 4"]
        box_colors = [(0, 200, 0), (200, 200, 0), (0, 200, 200), (200, 0, 200)]
        #zones=zones[::-1]  # Reversing the order of zones
        overlay_boxes = overlay.copy()
        for i, ((x1, y1), (x2, y2)) in enumerate(zones):
            cv2.rectangle(overlay_boxes, (x1, y1), (x2, y2), box_colors[i], 2)
            cv2.putText(overlay_boxes, zone_labels[i], (x1 + 10, y1 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, box_colors[i], 2) #added zone lines for better visibility
        cv2.addWeighted(overlay_boxes, 0.5, overlay, 0.7, 0, dst=overlay)

        count_text = f"Count: {int(total_count)}"
        (text_width, text_height), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        cv2.putText(overlay, count_text,(overlay.shape[1] - text_width - 10, overlay.shape[0] - 20), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)  #added text position

        #cv2.putText(overlay, f"Count: {int(total_count)}", (10, 40),
                    #cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        
        zone_count = get_zone_counts(density_map)
        zone_occupancy = get_zone_occupancy_percentage(density_map)
        send_json_data(zone_occupancy, zone_count)
        #print(zone_count)
        #print(zone_occupancy)

        
        ret, buffer = cv2.imencode('.jpg', overlay)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


if __name__ == "__main__":
    video_path='https://www.youtube.com/watch?v=7jlvN81lYJQ'  # Replace with your video URL or path
    #stream_url=convert_url(video_path)
    process_crowd_video(video_path)