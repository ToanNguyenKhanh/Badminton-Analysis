import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break
        frames.append(frame)
    cap.release()
    return frames # list of frame

def save_video(frames, ori_video_path, output_video_path):
    cap = cv2.VideoCapture(ori_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Writing frames onto output video
    for frame in frames:
        out.write(frame)
    out.release()

def convert_meters_to_pixels(frames):
    pass

