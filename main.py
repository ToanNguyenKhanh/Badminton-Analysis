import cv2
from utils import read_video, save_video
from tracker import PlayerTracker, ShuttleTracker


def main():
    ### input video ###
    input_video_path = 'test_video.mp4'
    # read video
    frames = read_video(input_video_path)

    # player detection

    player_tracker = PlayerTracker(model_path="train/player_output/models/weights/best.pt")
    player_detect = player_tracker.detect_player(frames, last_detect=True,
                                                    path_of_last_detect="last_detect/list_player_dict.pkl")

    # shuttle detection
    shuttle_tracker = ShuttleTracker(model_path="train/shuttle_output/models/weights/best.pt", )
    shuttle_detect = shuttle_tracker.detect_shuttle(frames, last_detect=True, path_of_last_detect="last_detect/list_shuttle_dict.pkl")
    shuttle_interpolate = shuttle_tracker.interpolate_shuttle_position(shuttle_detect)

    ### draw ###
    # draw player bbox
    output_frames = player_tracker.draw_player_bbox(frames, player_detect)

    # draw shuttle bbox
    output_frames = shuttle_tracker.draw_shuttle_bbox(output_frames, shuttle_interpolate)

    ### output video ###
    # save video
    output_video_path = 'output_video.mp4'
    save_video(output_frames, input_video_path, output_video_path)


if __name__ == '__main__':
    main()
