import cv2
import cvlib as cv
import numpy as np

# face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
# (H, W) = (None, None)
# id = 0
video = cv2.VideoWriter("rec_out.avi", cv2.VideoWriter_fourcc(*'MJPG'), 20, (frame.shape[1], frame.shape[0]))
while True:
    id = 0
    ret, frame = cap.read()
    # if W is None or H is None:
	# 	(H, W) = frame.shape[:2]
    rects = []
    face, confidence = cv.detect_face(frame)
    # print(face)

    for (i, (x, y, w, h)) in enumerate(face):
        # box = cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)

        # print(face[0][0])
        # print(face[0][2])
        # if face[0][0] >= [300]:
        #     print("face turing left")
        # if face[0][0] <= [250]:
        #     print("face turing right")
        # if face[0][0] >= [260] and face[0][0 ]<= [290]:
        #     print("Please turn the face left to right or right to left")
        #
        #     video.write(frame)
        # rects.append(box)



        # face_no = face.shape[0]
        # face_no +=1
        # box = face[0, 0, i, 3:7] * np.array([W, H, W, H])
    # objects = ct.update(rects)
    #
	# # loop over the tracked objects
	# for (objectID, centroid) in objects.items():
	# 	# draw both the ID of the object and the centroid of the
	# 	# object on the output frame
	# 	text = "{}".format(objectID)
	# 	cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
	# 		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # face_no += 1
        # sub_face = frame[y:h, x:w]
        # cv2.imshow("crop", sub_face)
        # label = "{}".format(id)
        # cv2.putText(frame, label, (x, y),  cv2.FONT_HERSHEY_SIMPLEX,
        #             0.7, (0, 255, 0), 2)
        # id += 1
        # print(a)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
cap.release()
cv2.destroyAllWindows()
