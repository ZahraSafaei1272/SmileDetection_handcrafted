import cv2
import joblib
import nbimporter
from smile_detection_preprocessing import resize_crop_img
from smile_detection_feature_extraction import extract_feature


model_name = 'svm_model.joblib'
scaler_name = 'scaler.joblib'
model = joblib.load(model_name)
scaler = joblib.load(scaler_name)
cap=cv2.VideoCapture(0)
while True:
    # Read frame from video capture
    ret, frame = cap.read()
    if ret:
    # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using a face detector
        face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_classifier.detectMultiScale(gray, 1.1, 4)

        # Loop over all detected faces
        for (x, y, w, h) in faces:
            # Extract features from the detected face using a pre-trained deep learning model
            # and reshape the features to match the input shape of the SVM classifier
            #size = (256, 256)           
            #resized_image = cv2.resize(frame[y:y+h, x:x+w], size)
            
            resized_image = resize_crop_img([frame[y:y+h, x:x+w]], 256, 256)
            test_features = extract_feature(resized_image)
            test_X = scaler.transform(test_features)

            prediction = model.predict(test_X)

            # If the SVM predicts a smile, draw a rectangle around the face and display a message
            if int(*prediction) == 1:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, 'Smiling', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Smile Detector', frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()            




