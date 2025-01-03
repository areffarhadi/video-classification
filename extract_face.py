import cv2
import os
import numpy as np
from facenet_pytorch import MTCNN

# Initialize MTCNN detector
detector = MTCNN()

def extract_faces_from_video(input_video_path, output_video_path, output_size=(224, 224)):

    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {input_video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Temporary storage for face frames
    face_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces in the frame
        boxes, _ = detector.detect(frame)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1) 

                # Extract face region
                face = frame[y1:y2, x1:x2]

                # Resize face region to the desired output size
                face_resized = cv2.resize(face, output_size)

                # Append the face frame to the list
                face_frames.append(face_resized)

    cap.release()

    if not face_frames:
        print("No faces detected in the video.")
        return

    # Write the face frames to a new video
    out = cv2.VideoWriter(output_video_path, fourcc, fps, output_size)

    for face_frame in face_frames:
        out.write(face_frame)

    out.release()
    print(f"Processed video saved to {output_video_path}")

# Example usage
input_folder = "AV_data/ADS"
output_folder = "AV_data/ADS_face"

os.makedirs(output_folder, exist_ok=True)

video_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]

total_videos = len(video_files)
processed_count = 0

for filename in video_files:
    processed_count += 1
    print(f"Processing video {processed_count}/{total_videos}: {filename}")
    input_video_path = os.path.join(input_folder, filename)
    output_video_path = os.path.join(output_folder, f"processed_{filename}")
    extract_faces_from_video(input_video_path, output_video_path)

    remaining = total_videos - processed_count
    print(f"Processed: {processed_count}, Remaining: {remaining}")

