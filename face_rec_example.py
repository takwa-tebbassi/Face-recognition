# CelebA Dataset used in this project
# http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

# For more information about the dataset, visit the project website:
# http://personal.ie.cuhk.edu.hk/~lz013/projects/CelebA.html


import face_recognition
import os  # to iterate over directories
import cv2  # to do image tests

KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
TOLERANCE = 0.6
FRAME_THIKNESS = 2
FONT_THINKNESS = 2
MODEL = "cnn"  #hog

print("loading known faces")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, name)

    if not os.path.isdir(person_dir):
        continue

    for filename in os.listdir(person_dir):
        filepath = os.path.join(person_dir, filename)

        image = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(image)

        if len(encodings) > 0:
            encoding = encodings[0]
            known_faces.append(encoding)
            known_names.append(name)
        else:
            print(f"[WARN] No face found in {filepath}")


print("processing unknown faces")

for filename in os.listdir(UNKNOWN_FACES_DIR):
    print(filename)
    image = face_recognition.load_image_file(f"{UNKNOWN_FACES_DIR}/{filename}")
    locations = face_recognition.face_locations(image, model= MODEL)
    encodings = face_recognition.face_encodings(image, locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found: {match}")

            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])

            color = [255, 0, 0]

            cv2.rectangle(image,top_left, bottom_right, color, FRAME_THIKNESS)

            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2]+22)
            cv2.rectangle(image,top_left, bottom_right, color, cv2.FILLED)

            cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), FONT_THINKNESS)

    cv2.imshow(filename, image)
    cv2.waitKey(1000)  # 1 sec
    #cv2.destroyWindow(filename)
