"""

MIT License

Copyright 2024 Maximilian Petersson and Nahom Solomon

"""



from ultralytics import YOLO
from cv2 import VideoCapture, VideoWriter, VideoWriter_fourcc
import time
import sys

model = YOLO('/content/last.pt')
#TRACKER = 'botsort.yaml'
TRACKER = 'bytetrack.yaml'

iterations = 11 #First iteration is warmup.

cumulative_min_fps = 0.0
cumulative_average_fps = 0.0
cumulative_max_fps = 0.0

for i in range(iterations):
  cap = VideoCapture('/content/v1imgz640 - TrimFINAL.mp4')
  out = VideoWriter('output.avi', VideoWriter_fourcc(*'XVID'), 30.0, (640, 640))
  
  frame_counter = 0

  min_fps = sys.float_info.max
  cumulative_fps = 0.0
  max_fps = sys.float_info.min

  while True:
    ret, frame = cap.read()
    if not ret:
      break

    start = time.time()
    output = model.track(frame, tracker=TRACKER, conf=0.257, persist=True, show=False)
    end = time.time()

    frame_counter += 1

    fps = 1.0 / (end - start)
    print("%.1f" % fps)

    min_fps = min(min_fps, fps)
    cumulative_fps += fps
    max_fps = max(max_fps, fps)

  if(i != 0): #First iteration is warmup.
    cumulative_min_fps += min_fps
    cumulative_average_fps += (cumulative_fps / frame_counter)
    cumulative_max_fps += max_fps

    cap.release()
    out.release()

print("Min FPS: %.1f" % (cumulative_min_fps / iterations))
print("Average FPS: %.1f" % (cumulative_average_fps / iterations))
print("Max FPS: %.1f" % (cumulative_max_fps / iterations))
