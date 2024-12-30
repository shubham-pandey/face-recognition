/*
compilation command
clang++ -std=c++17 -o face3 newface.cpp $(pkg-config --cflags --libs opencv4)
*/
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <iostream>
#include <cstdlib>  // For system command
#include <chrono>   // For time calculation
#include <vector>   // For vector
#include <sys/types.h>
#include <unistd.h> // For fork and exec

using namespace std;
using namespace cv;

// Function to pause media playback using AppleScript
void pauseMedia() {
    pid_t pid = fork();
    if (pid == 0) { // Child process
        execl("/usr/bin/osascript", "osascript", "-e", "tell application \"System Events\" to key code 49", (char *)NULL);
        exit(0); // Exit if exec fails
    }
}

// Function to play media playback using AppleScript
void playMedia() {
    pid_t pid = fork();
    if (pid == 0) { // Child process
        execl("/usr/bin/osascript", "osascript", "-e", "tell application \"System Events\" to key code 49", (char *)NULL);
        exit(0); // Exit if exec fails
    }
}

int main() {
    // Load the Haar cascade for face detection
    CascadeClassifier faceCascade;
    if (!faceCascade.load("haarcascade_frontalface_default.xml")) {
        cout << "Error loading face cascade\n";
        return -1;
    }

    // Start video capture from the webcam
    VideoCapture cap(0); // 0 is the default camera
    if (!cap.isOpened()) {
        cout << "Error opening video stream\n";
        return -1;
    }

    // Threshold for minimum face size
    const int faceSizeThreshold = 20000;

    // Timers for face detection and no face detection
    auto lastFaceDetectedTime = chrono::steady_clock::now();
    auto lastSmallFaceTime = chrono::steady_clock::now();
    auto lastNoFaceTime = chrono::steady_clock::now();
    bool faceDetected = false;
    bool mediaPaused = false;
    bool smallFaceDetected = false;

    while (true) {
        Mat frame;
        cap >> frame; // Capture the current frame

        if (frame.empty()) {
            cout << "Error capturing frame\n";
            break;
        }

        // Convert frame to grayscale
        Mat grayFrame;
        cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
        equalizeHist(grayFrame, grayFrame); // Equalize histogram for better detection

        // Detect faces
        vector<Rect> faces;
        faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        if (faces.empty()) {
            if (faceDetected) {
                lastNoFaceTime = chrono::steady_clock::now();
                faceDetected = false;
            }

            auto now = chrono::steady_clock::now();
            auto duration = chrono::duration_cast<chrono::seconds>(now - lastNoFaceTime);

            if (duration.count() >= 1 && !mediaPaused) {
                cout << "Pausing media due to no face detected...\n";
                pauseMedia();
                mediaPaused = true;
            }
        } else {
            // A face is detected
            if (!faceDetected) {
                lastFaceDetectedTime = chrono::steady_clock::now();
                faceDetected = true;
            }

            auto now = chrono::steady_clock::now();
            auto duration = chrono::duration_cast<chrono::seconds>(now - lastFaceDetectedTime);

            // Check the size of the first detected face
            Rect face = faces[0];
            int faceSize = face.width * face.height;

            cout << "Detected face size: " << faceSize << "\n";

            if (faceSize < faceSizeThreshold) {
                if (!smallFaceDetected) {
                    lastSmallFaceTime = chrono::steady_clock::now();
                    smallFaceDetected = true;
                }

                auto smallFaceDuration = chrono::duration_cast<chrono::seconds>(now - lastSmallFaceTime);
                if (smallFaceDuration.count() >= 1 && !mediaPaused) {
                    cout << "Pausing media due to small face size detected for 2 seconds...\n";
                    pauseMedia();
                    mediaPaused = true;
                }
            } else {
                smallFaceDetected = false; // Reset small face detection timer

                if (duration.count() >= 1 && mediaPaused) {
                    cout << "Resuming media as face is continuously detected with sufficient size...\n";
                    playMedia();
                    mediaPaused = false;
                }
            }

            // Draw rectangle around the face
            rectangle(frame, face, Scalar(255, 0, 0), 2);
        }

        // Show the frame with detections
        imshow("Face Detection", frame);

        // Break the loop if the user presses 'q'
        if (waitKey(10) == 'q') {
            break;
        }
    }

    cap.release(); // Release the camera
    destroyAllWindows(); // Close all OpenCV windows
    return 0;
}
