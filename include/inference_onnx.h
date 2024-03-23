#ifndef INFERENCE_H
#define INFERENCE_H

// Cpp native
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

namespace YOLO
{
    struct Detection
    {
        int class_id{0};
        std::string className{};
        float confidence{0.0};
        cv::Scalar color{};
        cv::Rect box{};
        cv::Mat boxMask;
    };

    class ConeDetector
    {
    public:
        ConeDetector(const std::string &onnxModelPath, const cv::Size &modelInputShape = {640, 640}, const std::string &classesTxtFile = "", const bool &runWithCuda = true);
        std::vector<Detection> Infer(cv::Mat &srcImg);

    private:
        void loadOnnxNetwork();
        cv::Mat GetMask(const cv::Mat &maskProposal, const cv::Mat &mask_protos, cv::Rect &temp_rect, cv::Size src_img_shape, cv::Vec4d params);

        std::string modelPath{};
        std::string classesPath{};
        bool cudaEnabled{};

        std::vector<std::string> classes{"seg_blue_cone", "seg_large_orange_cone", "seg_orange_cone", "seg_unknown_cone", "seg_yellow_cone"};
        std::vector<cv::Scalar> colors{cv::Scalar(190, 100, 20), cv::Scalar(0, 110, 255), cv::Scalar(0, 110, 255), cv::Scalar(127, 127, 127), cv::Scalar(60, 255, 255)};

        cv::Size2f modelShape{};

        template <typename T>
        T VectorProduct(const std::vector<T>& v)
        {
            return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
        };


        float modelConfidenceThreshold {0.25};
        float modelScoreThreshold      {0.45};
        float modelNMSThreshold        {0.50};

        bool letterBoxForSquare = true;

        Ort::Env ort_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "yolov8_seg");
        Ort::SessionOptions ort_session_opt;
        // Ort::Session session(env, onnxModelPath, session_options);
        Ort::Session* ort_session = nullptr;
        Ort::AllocatorWithDefaultOptions allocator;
	    Ort::MemoryInfo ort_memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        std::shared_ptr<char> inputName, output_name0, output_name1;
        std::vector<char*> inputNodeNames;
	    std::vector<char*> outputNodeNames;
        size_t input_tensor_length;
        cv::Size input_size;

        std::vector<long int> inputTensorShape;
        std::vector<long int> outputTensorShape;
        std::vector<long int> outputMaskTensorShape;


        cv::dnn::Net net;
    };
}

#endif // INFERENCE_H
