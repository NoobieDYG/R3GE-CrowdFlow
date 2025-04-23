import numpy as np
import cv2 as cv
import os
os.environ["PAFY_BACKEND"] = "internal"
import pafy
from yt_dlp import YoutubeDL
import time
from ultralytics import YOLO

model= YOLO('yolov8n.pt')

def convert_url(yt_url):
    ydl_opts={
        'format': 'best[ext=mp4]/best',
        'quiet': True,
        'noplaylist': True,  
          }
    with YoutubeDL(ydl_opts) as ydl:
        info=ydl.extract_info(yt_url, download=False)
        return info['url']

'''def video_process(video_path):
    cap= cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        zones=divide_into_zones(frame, grid_size=(3, 3))
        counts=get_crowd_counts_yolo(frame, zones)
        output_frame= overlay_heatmap(frame, zones, counts)
        #gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

       
        cv.imshow('Grayscale Video', output_frame)

       
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

   
    cap.release()
    cv.destroyAllWindows()

def display_heat_video(video_path):
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        zones=divide_into_zones(frame, grid_size=(3, 3))
        counts=get_crowd_counts_yolo(frame, zones)
        output_frame= overlay_heatmap(frame, zones, counts)
        heatmap_frame = cv.applyColorMap(frame, cv.COLORMAP_JET)
        cv.drawContours(heatmap_frame, [np.array([[0, 0], [0, 100], [100, 100], [100, 0]])], -1, (0, 255, 0), 2)

        cv.imshow('Heatmap Video', heatmap_frame)
        
        print(counts)
        
        time.sleep(0.03)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

def divide_into_zones(frame, grid_size=(3, 3)):
    h, w = frame.shape[:2]
    zone_height = h // grid_size[0]
    zone_width = w // grid_size[1]

    zones = []
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            top_left = (j * zone_width, i * zone_height)
            bottom_right = ((j + 1) * zone_width, (i + 1) * zone_height)
            zones.append((top_left, bottom_right))
    return zones

def count_to_color(count, max_count=10):
    ratio = min(count / max_count, 1.0)
    
    color = (0, int((1 - ratio) * 255), int(ratio * 255))  
    return color

def overlay_heatmap(frame, zones, counts):
    heatmap = frame.copy()
    for (zone, count) in zip(zones, counts):
        color = count_to_color(count)
        cv.rectangle(heatmap, zone[0], zone[1], color, -1)

    
    blended = cv.addWeighted(frame, 0.6, heatmap, 0.4, 0)
    return blended


def get_crowd_counts_yolo(frame, zones):
    results = model(frame)[0]
    counts = [0] * len(zones)

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id != 0:
            continue  # not a person
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Assign person to zone
        for i, (tl, br) in enumerate(zones):
            if tl[0] <= cx <= br[0] and tl[1] <= cy <= br[1]:
                counts[i] += 1
                break

    return counts'''

if __name__ == "__main__":
    #video_path="C:\\Users\\Affaan Jaweed\\Desktop\\crowd_control_hackazrd\\vision_model\\dataset\\umrah_test_1.mp4"
    video_path='https://www.youtube.com/watch?v=uvJEvjk7agA'
    stream_url=convert_url(video_path)

    #video_process(video_path)
    #display_heat_video(stream_url)  