import cv2
import numpy as np
import os

def extract_motion_vectors(video_path, max_corners = 100):
    """
    Extracts motion vectors from a video using Lucas-Kanade optical flow
    with Shi-Tomasi corner detection.

    Args:
        video_path (str): Path to the video file.
        max_corners (int): Maximum number of corners to save in each frame.

    Returns:
        list: A list of numpy arrays, each containing flow vectors for a pair of frames, or an empty list if no corners were detected.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")

    motion_vectors = []
    prev_gray = None

    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=1000,
                          qualityLevel=0.01,
                          minDistance=0.1)

    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15),
                   maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            # Find good corners using Shi-Tomasi
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

            if p0 is not None:
                # Calculate optical flow
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)
                # Select good points
                if p1 is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    # Calculate flow vectors by subtracting new points from old points
                    flow_vectors = good_new - good_old

                    # Pad or truncate flow_vectors to have consistent length
                    if len(flow_vectors) > max_corners:
                       flow_vectors = flow_vectors[:max_corners]
                    elif len(flow_vectors) < max_corners:
                        padding = np.zeros((max_corners - len(flow_vectors), 2))
                        flow_vectors = np.concatenate((flow_vectors, padding))

                    motion_vectors.append(flow_vectors)
            else:
                print("No corners detected in the previous frame. Skipping optical flow calculation for this pair of frames.")
        prev_gray = gray

    cap.release()
    return motion_vectors

def save_motion_vectors(motion_vectors, output_file_path):
    """
    Saves the extracted motion vectors to a .npy file.

    Args:
        motion_vectors (list): List of motion vector numpy arrays
        output_file_path (str): Output path where to save the numpy array

    """
    np.save(output_file_path, np.array(motion_vectors))

if __name__ == "__main__":
    video_path = input("Enter the path to your video file: ")
    try:
        motion_data = extract_motion_vectors(video_path)

        # Get the name of the video file
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # Construct output filename with .npy extension in the same directory
        output_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{video_name}_vectors.npy")
        save_motion_vectors(motion_data, output_file_path)
        print(f"Successfully extracted and saved motion vectors to: {output_file_path}")

    except Exception as e:
        print(f"Error extracting and saving motion vectors: {e}")