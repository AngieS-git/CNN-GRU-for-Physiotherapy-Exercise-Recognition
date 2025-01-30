import cv2
import numpy as np
import os

def extract_motion_vectors(video_path):
    """
    Extracts motion vectors from a video using Lucas-Kanade optical flow
    with Shi-Tomasi corner detection.

    Args:
        video_path (str): Path to the video file.

    Returns:
        list: A list of numpy arrays, each containing flow vectors.
    """
    if not os.path.exists(video_path):
        raise IOError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")

    motion_vectors = []
    prev_gray = None

    # Parameters for Shi-Tomasi corner detection
    feature_params = dict(maxCorners=500,
                          qualityLevel=0.3,
                          minDistance=7,
                          blockSize=7)

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
                     
                    # Calculate flow vectors by substracting new points to old points
                    flow_vectors = good_new - good_old
                    
                    # Append current flow vectors to list
                    motion_vectors.append(flow_vectors)
            else:
                print("No corners detected in the previous frame. Skipping optical flow calculation for this pair of frames.")
            
        prev_gray = gray

    cap.release()
    return motion_vectors

if __name__ == "__main__":
    video_path = input("Enter the path to your video file: ")  # Get video path from user
    try:
        motion_data = extract_motion_vectors(video_path)
        print(f"Successfully extracted motion vectors from: {video_path}")

        # Optional - you can print or inspect the motion vectors here
        # Example:
        for i, vectors in enumerate(motion_data):
            print(f"Frame {i}: {len(vectors)} motion vectors detected")
            # print(vectors) #Uncomment to see the motion vectors values

    except Exception as e:
         print(f"Error extracting motion vectors: {e}")