import cv2
import dlib
import keras
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.utils.generic_utils import CustomObjectScope


IMAGE_SIZE=224

def prepare_image(img):
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

def predict(img):
    """
    Predict face crop from frame
    :param img:
    :return: If boss is appear when open the code IDE
    """
    try:
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        probs = model.predict(prepare_image(img))
        is_boss = np.argmax(probs[0])
        return is_boss
    except:
        return False
    
with CustomObjectScope({'relu6': keras.layers.ReLU,'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
    model = load_model('model_save.h5')

cap = cv2.VideoCapture('bandicam 2019-03-17 13-37-11-071.mp4')
# initialize hog + svm based face detector
hog_face_detector = dlib.get_frontal_face_detector()
boss_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Resize window
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', 400, 400)
        # Resize frame for decrease predict time
        scale = 0.5
        resize_frame = cv2.resize(frame, (int(frame.shape[1]*scale), int(frame.shape[0]*scale)))
        faces_hog = hog_face_detector(resize_frame)
        frame_h, frame_w, _ = frame.shape
        reindex_x = lambda x: max(min(x, frame_w), 1)
        reindex_y = lambda x: max(min(x, frame_h), 1)

        # loop over detected faces
        for face in faces_hog:
            x = reindex_x(int(face.left() / scale))
            y = reindex_y(int(face.top() / scale))
            r = reindex_x(int(face.right() / scale))
            b = reindex_y(int(face.bottom() / scale))

            # draw box over face
            crop_face = frame[x: r, y: b]
            is_boss = predict(crop_face)
            if is_boss:
                cv2.putText(frame, "BOSS", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 0, 255), 4)
                cv2.rectangle(frame, (x, y), (r, b), (0, 0, 255), 2)
                boss_count += 1
                # Open your IDE application for coding
                if boss_count > 3:
                    # os.system('open -a "PyCharm CE"')
                    print('I saw boss')
                    boss_count = 0
            else:
                cv2.putText(frame, "NORMAL", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 255, 0), 4)
                cv2.rectangle(frame, (x, y), (r, b), (0, 255, 0), 2)


        cv2.imshow("frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()