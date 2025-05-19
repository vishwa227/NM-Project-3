import cv2
import numpy as np

def order_points(pts):
    # Orders points in the order: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def find_aruco_markers(frame, aruco_dict_type=cv2.aruco.DICT_4X4_50):
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    aruco_params = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
    return corners, ids

def main():
    # Load the image to overlay (your AR content)
    source_img = cv2.imread("source.jpg")  # Replace with your image path
    if source_img is None:
        print("Source image not found!")
        return

    # Define the IDs of the four markers at the corners (adjust as per your setup)
    desired_ids = [0, 1, 2, 3]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corners, ids = find_aruco_markers(frame)
        output = frame.copy()

        if ids is not None and len(ids) >= 4:
            ids = ids.flatten()
            # Check if all desired IDs are detected
            if all(marker_id in ids for marker_id in desired_ids):
                marker_corners = []
                for marker_id in desired_ids:
                    idx = np.where(ids == marker_id)[0][0]
                    marker_corners.append(corners[idx][0][0])  # Use the top-left corner for each marker

                marker_corners = np.array(marker_corners, dtype="float32")
                ordered_corners = order_points(marker_corners)

                (h, w) = source_img.shape[:2]
                src_pts = np.array([
                    [0, 0],
                    [w - 1, 0],
                    [w - 1, h - 1],
                    [0, h - 1]
                ], dtype="float32")
                dst_pts = ordered_corners

                # Warp perspective
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped = cv2.warpPerspective(source_img, M, (frame.shape[1], frame.shape[0]))

                # Masking and blending
                mask = np.zeros((frame.shape[0], frame.shape[1]), dtype="uint8")
                cv2.fillConvexPoly(mask, dst_pts.astype("int32"), 255)
                mask_inv = cv2.bitwise_not(mask)
                frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
                warped_fg = cv2.bitwise_and(warped, warped, mask=mask)
                output = cv2.add(frame_bg, warped_fg)

        cv2.imshow("Augmented Reality", output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()