import numpy as np
import cv2
import os
import time
from yt_dlp import YoutubeDL

def manual_density_heatmap(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
   
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    dist = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    
    density_map = cv2.GaussianBlur(dist, (101, 101), 30)
    
    return density_map

def enhance_contrast(heatmap):
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(heatmap)
    
    
    enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    _, high_density = cv2.threshold(enhanced, 180, 255, cv2.THRESH_BINARY)
    high_density = cv2.GaussianBlur(high_density, (31, 31), 0)
    
    
    result = cv2.addWeighted(enhanced, 0.7, high_density, 0.3, 0)
    
    return result

def process_frame(frame):
    
    density_map = manual_density_heatmap(frame)
    
    
    enhanced_map = enhance_contrast(density_map)
    
    
    colored_heatmap = cv2.applyColorMap(enhanced_map, cv2.COLORMAP_COOL)
    
    
    result = cv2.addWeighted(frame, 0.5, colored_heatmap, 0.5, 0)
    
    return result

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video dimensions: {width}x{height}, FPS: {fps}")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        print(f"Processing frame {frame_count}")
        
        if frame_count % 3 == 0 or frame_count == 1:
            processed_frame = process_frame(frame)
            cv2.namedWindow('High Contrast Density Heatmap', cv2.WINDOW_NORMAL)
            cv2.imshow('High Contrast Density Heatmap', processed_frame)
            time.sleep(0.03)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(0)
            
    cap.release()
    cv2.destroyAllWindows()

def convert_url(yt_url):
    ydl_opts={
        'format': 'best[ext=mp4]/best',
        'quiet': True,
        'noplaylist': True,  
          }
    with YoutubeDL(ydl_opts) as ydl:
        info=ydl.extract_info(yt_url, download=False)
        return info['url']
    
if __name__ == "__main__":
    video_path='https://www.youtube.com/watch?v=LXe3_hBybsc'
    stream_url=convert_url(video_path)
    process_video(stream_url)