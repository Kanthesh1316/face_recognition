import cv2
from simple_facerec import SimpleFacerec

def main():
    # Initialize the SimpleFacerec class
    sfr = SimpleFacerec()
    sfr.load_encoding_images("images/")  # Load known face images

    # Load Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not accessible.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        # Detect Faces
        face_locations, face_names = sfr.detect_known_faces(frame)
        
        # Debugging: print number of faces detected
        print(f"Detected faces: {len(face_locations)}")

        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc
            
            # Color coding for known (green) and unknown (red) faces
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)  # Draw rectangle around face

            # Display the name above the rectangle
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)

        cv2.imshow("frames", frame)

        key = cv2.waitKey(1)
        if key == 27:  # Press 'Esc' key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
