#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main() {
    // Load the Haar cascades for face and eye detection
    CascadeClassifier faceCascade;
    CascadeClassifier eyeCascade;
    
    if (!faceCascade.load("haarcascade_frontalface_default.xml")) {
        cout << "Error loading face cascade\n";
        return -1;
    }
    
    if (!eyeCascade.load("haarcascade_eye.xml")) {
        cout << "Error loading eye cascade\n";
        return -1;
    }

    // Start video capture from webcam
    VideoCapture cap(0); // 0 is the default camera
    if (!cap.isOpened()) {
        cout << "Error opening video stream\n";
        return -1;
    }

    while (true) {
        Mat frame;
        cap >> frame;  // Capture the current frame

        if (frame.empty()) {
            cout << "Error capturing frame\n";
            break;
        }

        // Convert frame to grayscale
        Mat grayFrame;
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
        equalizeHist(grayFrame, grayFrame);  // Equalize histogram for better detection

        // Detect faces
        vector<Rect> faces;
        faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        for (size_t i = 0; i < faces.size(); i++) {
            // Draw rectangle around the face
            rectangle(frame, faces[i], Scalar(255, 0, 0), 2);

            Mat faceROI = grayFrame(faces[i]);

            // Detect eyes within the face region
            vector<Rect> eyes;
            eyeCascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

            for (size_t j = 0; j < eyes.size(); j++) {
                // Get the coordinates of the eye within the face ROI
                Rect eye = eyes[j];
                Point eyeCenter(faces[i].x + eye.x + eye.width / 2, faces[i].y + eye.y + eye.height / 2);
                int radius = cvRound((eye.width + eye.height) * 0.25);
                circle(frame, eyeCenter, radius, Scalar(0, 255, 0), 2);
            }
        }

        // Show the frame with detections
        imshow("Eye Tracking", frame);

        // Break the loop if the user presses 'q'
        if (waitKey(10) == 'q') {
            break;
        }
    }

    cap.release(); // Release the camera
    destroyAllWindows(); // Close all OpenCV windows
    return 0;
}