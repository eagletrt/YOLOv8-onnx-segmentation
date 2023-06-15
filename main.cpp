#include <iostream>
#include <vector>
#include <chrono>
#include <getopt.h>

#include <opencv2/opencv.hpp>

#include "inference.h"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    std::string projectBasePath = "/mnt/d/Projects/Eagle/ultralytics"; // Set your ultralytics base path
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::milliseconds;

    bool runOnGPU = true;

    //
    // Pass in either:
    //
    // "yolov8s.onnx" or "yolov5s.onnx"
    //
    // To run Inference with yolov8/yolov5 (ONNX)
    //

    // Note that in this example the classes are hard-coded and 'classes.txt' is a place holder.
    YOLO::Inference inf(projectBasePath + "/best-seg-640-480.onnx", cv::Size(480, 640), "classes.txt", runOnGPU);

    cv::VideoCapture capture(projectBasePath + "/test.mp4");
    cv::Mat frame;

    //cv::Mat frame = cv::imread(projectBasePath + "/ultralytics/assets/bus.jpg");

    if( !capture.isOpened() )
        std::cout <<  "Error when reading steam_avi" << std::endl;
        //return -1;

    auto t1 = high_resolution_clock::now();
    for( ; ; )
    {
        capture >> frame;
        if (frame.empty())
            break;
        cv::resize(frame, frame, cv::Size(480, 640));
        if (frame.empty())
            break;

        // Inference starts here...
        
        std::vector<YOLO::Detection> output = inf.runInference(frame);

        int detections = output.size();
        //std::cout << "Number of detections:" << detections << std::endl;

        for (int i = 0; i < detections; ++i)
        {
            YOLO::Detection detection = output[i];

            cv::Rect box = detection.box;
            cv::Scalar color = detection.color;
            cv::Mat mask = detection.boxMask;

            // Detection box
            cv::rectangle(frame, box, color, 2);

            // Detection mask
            frame(box).setTo(color, mask);

            // Detection box text
            std::string classString = detection.className + ' ' + std::to_string(detection.confidence).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

            cv::rectangle(frame, textBox, color, cv::FILLED);
            cv::putText(frame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
        }
        // Inference ends here...

        // This is only for preview purposes
        float scale = 1;
        cv::resize(frame, frame, cv::Size(frame.cols*scale, frame.rows*scale));
        cv::imshow("Inference", frame);

        cv::waitKey(1);
    }
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << ms_double.count() << "ms\n";
}
