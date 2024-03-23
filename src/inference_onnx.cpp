#include "inference_onnx.h"
#include <onnxruntime_cxx_api.h>
namespace YOLO
{
    ConeDetector::ConeDetector(const std::string &onnxModelPath, const cv::Size &modelInputShape, const std::string &classesTxtFile, const bool &runWithCuda)
    {
        modelPath = onnxModelPath;
        modelShape = modelInputShape;
        classesPath = classesTxtFile;
        cudaEnabled = runWithCuda;

        ort_session = new Ort::Session(ort_env, onnxModelPath.c_str(), ort_session_opt);

        //init input
        inputName = std::move(ort_session->GetInputNameAllocated(0, allocator));
		inputNodeNames.push_back(inputName.get());
        /*
        for (const char *str : inputNodeNames)
        {
            std::cout << str << std::endl;
        }
        */      
        Ort::TypeInfo inputTypeInfo = ort_session->GetInputTypeInfo(0);
        inputTensorShape = inputTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
        input_tensor_length = VectorProduct(inputTensorShape);
        input_size = cv::Size(inputTensorShape[3], inputTensorShape[2]);

        //init output
        size_t outputNodesNum = ort_session->GetOutputCount();
        
        std::cout << "This model has " << outputNodesNum << " outputs";
        if (outputNodesNum != 2) {
            std::cout << ", which is not a segmentation model. Please check your model name or path!" << std::endl;
        }
        else {
            std::cout << ", everything good!" << std::endl;
        }
        
        output_name0 = std::move(ort_session->GetOutputNameAllocated(0, allocator));
        output_name1 = std::move(ort_session->GetOutputNameAllocated(1, allocator));
        Ort::TypeInfo type_info_output0(nullptr);
        Ort::TypeInfo type_info_output1(nullptr);
        bool flag = false;
        flag = strcmp(output_name0.get(), output_name1.get()) < 0;
        if (flag)  //make sure "output0" is in front of  "output1"
        {
            type_info_output0 = ort_session->GetOutputTypeInfo(0);  //output0
            type_info_output1 = ort_session->GetOutputTypeInfo(1);  //output1
            outputNodeNames.push_back(output_name0.get());
            outputNodeNames.push_back(output_name1.get());

        }
        else {
            type_info_output0 = ort_session->GetOutputTypeInfo(1);  //output0
            type_info_output1 = ort_session->GetOutputTypeInfo(0);  //output1
            outputNodeNames.push_back(output_name1.get());
            outputNodeNames.push_back(output_name0.get());
        }

        outputTensorShape = type_info_output0.GetTensorTypeAndShapeInfo().GetShape();
        outputMaskTensorShape = type_info_output1.GetTensorTypeAndShapeInfo().GetShape();

        for (const char *str : outputNodeNames)
        {
            std::cout << str << std::endl;
        }
        /*
        //warm up
        if (isCuda && warmUp) {
            //draw run
            cout << "Start warming up" << endl;
            size_t input_tensor_length = VectorProduct(_inputTensorShape);
            float* temp = new float[input_tensor_length];
            std::vector<Ort::Value> input_tensors;
            std::vector<Ort::Value> output_tensors;
            input_tensors.push_back(Ort::Value::CreateTensor<float>(
                _OrtMemoryInfo, temp, input_tensor_length, _inputTensorShape.data(),
                _inputTensorShape.size()));
            for (int i = 0; i < 3; ++i) {
                output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
                    _inputNodeNames.data(),
                    input_tensors.data(),
                    _inputNodeNames.size(),
                    _outputNodeNames.data(),
                    _outputNodeNames.size());
            }

            delete[]temp;
        }
        */
        // loadOnnxNetwork();
        // loadClassesFromFile(); The classes are hard-coded for this example
    }

    std::vector<Detection> ConeDetector::Infer(cv::Mat& srcImg)
    {
        cv::Mat borderImg = srcImg;
        cv::Size image_size = srcImg.size();
        cv::Vec4d temp_param = {1,1,0,0};
        bool keep_ratio = true;
        if (image_size != input_size)
        {
            if(keep_ratio){
                float ratio = std::min((float)input_size.height / image_size.height, (float)input_size.width / image_size.width);
                cv::resize(borderImg, borderImg, cv::Size(), ratio, ratio);
                float pad_h = (input_size.height - borderImg.rows) / 2;
                float pad_w = (input_size.width - borderImg.cols) / 2;
                cv::copyMakeBorder(borderImg, borderImg, pad_h, pad_h, pad_w, pad_w, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
                temp_param[0] = 1 / ratio;
                temp_param[1] = 1 / ratio;
                temp_param[2] = pad_w;
                temp_param[3] = pad_h;
            }
            else{
                float ratio_x = (float)input_size.width / image_size.width;
                float ratio_y = (float)input_size.height / image_size.height;
                cv::resize(borderImg, borderImg, cv::Size(), ratio_x, ratio_y);
                temp_param[0] = 1 / ratio_x;
                temp_param[1] = 1 / ratio_y;
            }
        }

        std::vector<cv::Mat> input_images;
        std::vector<cv::Vec4d> params;
        input_images.push_back(borderImg);
        params.push_back(temp_param);
        image_size = borderImg.size();

        std::vector<std::vector<Detection>> temp_output;
        cv::Mat blob = cv::dnn::blobFromImage(borderImg, 1.0/255.0, input_size, cv::Scalar(), true, false);
        /*
        net.setInput(blob);
        std::vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());
        */
        // Run inference
        
        Ort::Value input_tensors = Ort::Value::CreateTensor<float>(ort_memory_info, (float*)blob.data, input_tensor_length, inputTensorShape.data(), inputTensorShape.size());

        Ort::RunOptions run_options;

        std::vector<Ort::Value> output_tensors = ort_session->Run(run_options, inputNodeNames.data(), &input_tensors, inputNodeNames.size(), outputNodeNames.data(), outputNodeNames.size());

        std::vector<cv::Mat> outputs;
        std::vector<long int> shape_0_long = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
        std::vector<int> shape_0(begin(shape_0_long), end(shape_0_long));
        std::vector<long int> shape_1_long = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();
        ONNXTensorElementDataType asd = output_tensors[1].GetTensorTypeAndShapeInfo().GetElementType();
        std::vector<int> shape_1(begin(shape_1_long), end(shape_1_long));

        cv::Mat outputs_0(shape_0, CV_32F, (float*)output_tensors[0].GetTensorMutableRawData());
        cv::Mat outputs_1(shape_1, CV_32F, (float*)output_tensors[1].GetTensorMutableRawData());
       
        int rows = outputs_0.size[1];
        int dimensions = outputs_0.size[2];
        //std::cout << outputs_1.size << std::endl;
        std::vector<int> mask_protos_shape = {(int)outputs_1.size[0],(int)outputs_1.size[1],(int)outputs_1.size[2],(int)outputs_1.size[3] };
        int mask_protos_length = outputs_1.size[0]*outputs_1.size[1]*outputs_1.size[2]*outputs_1.size[3];
        
        //std::cout << outputs_0.size << std::endl;
        //std::cout << outputs_1.size << std::endl;
        rows = outputs_0.size[2];
        dimensions = outputs_0.size[1];

        outputs_0 = outputs_0.reshape(1, dimensions);
        cv::transpose(outputs_0, outputs_0);
        
        float *data = (float *)outputs_0.data;

        float x_factor = (float)image_size.width / input_size.width;
        float y_factor = (float)image_size.height / input_size.height;

        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        std::vector<std::vector<float>> picked_proposals;
        for (int i = 0; i < rows; ++i)
        {
            float *classes_scores = data+4;
            std::vector<float> mask_point(data + 4 + classes.size(), data + dimensions);
        

            cv::Mat scores(1, classes.size(), CV_32FC1, classes_scores);
            //std::cout << scores << std::endl;
            cv::Point class_id;
            double maxClassScore;

            minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);

            if (maxClassScore > modelScoreThreshold)
            {
                confidences.push_back(maxClassScore);
                class_ids.push_back(class_id.x);
                picked_proposals.push_back(mask_point);
                //net_width == dimensions
                //segChannels == number of elements of sementation in net_width == 16
                //net_height == rows
                
                //cv::Mat data2(1, dimensions, CV_32FC1, data);
                //std::cout << data2 << std::endl;

                float x = data[0];
                float y = data[1];
                float w = data[2];
                float h = data[3];

                //std::cout << "size: " << x << " " << y << " " << w << " "  << h << std::endl;
                
                int left = MAX(int(x - 0.5 * w) * x_factor, 0);
                int top = MAX(int(y - 0.5 * h) * y_factor, 0);

                //std::cout << left << " " << top << std::endl;
                //std::cout << x_factor << " " << y_factor << std::endl;

                int width = MIN(int(w * x_factor), image_size.width - left);
                int height = MIN(int(h * y_factor), image_size.height - top);
                //std::cout << "size rect: " << left << " " << top << " " << width << " "  << height << std::endl;
                //std::cout << "size boundaries: " << left << " " << top << " " << left+width << " "  << top+height << std::endl;
                boxes.push_back(cv::Rect(left, top, width, height));
            }
            
            data += dimensions;
        }
        std::vector<int> nms_result;
        cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);
        std::vector<Detection> detections{};
        //std::vector<std::vector<float>> temp_mask_proposals;
        for (unsigned long i = 0; i < nms_result.size(); ++i)
        {
            int idx = nms_result[i];

            Detection result;
            result.class_id = class_ids[idx];
            result.confidence = confidences[idx];
            //temp_mask_proposals.push_back(picked_proposals[idx]);
            
            /*
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int> dis(100, 255);
            result.color = cv::Scalar(dis(gen),
                                    dis(gen),
                                    dis(gen));
            */
            result.color = colors[result.class_id];

            result.className = classes[result.class_id];
            result.box = cv::Rect((boxes[idx].x - temp_param[2]) * temp_param[0], (boxes[idx].y - temp_param[3]) * temp_param[1], boxes[idx].width * temp_param[0], boxes[idx].height * temp_param[1]);
            //result.box = boxes[idx];
            //std::cout << boxes[idx].x << " " << boxes[idx].y << " " << boxes[idx].width << " " << boxes[idx].height << std::endl;

            result.boxMask = ConeDetector::GetMask(cv::Mat(picked_proposals[idx]), outputs_1, boxes[idx], image_size, temp_param);
            cv::resize(result.boxMask, result.boxMask, cv::Size(result.box.width, result.box.height));
            detections.push_back(result);
        }
        return detections;
    }

    cv::Mat ConeDetector::GetMask(const cv::Mat& maskProposal, const cv::Mat& mask_protos, cv::Rect& temp_rect, cv::Size src_img_shape, cv::Vec4d params) {
        long int net_height = inputTensorShape[2];
        long int net_width = inputTensorShape[3];

        long int seg_channels = outputMaskTensorShape[1];
        long int seg_height = outputMaskTensorShape[2];
        long int seg_width = outputMaskTensorShape[3];

        //std::cout << net_height << " " << net_width << " " << seg_channels << " "  << seg_height << " " << seg_width << std::endl;
        // crop from mask_protos
        int rang_x = floor((temp_rect.x / float(net_width)) * seg_width);
        int rang_y = floor((temp_rect.y / float(net_height)) * seg_height);
        int rang_w = ceil((((temp_rect.x + temp_rect.width)) / float(net_width)) * seg_width) - rang_x;
        int rang_h = ceil((((temp_rect.y + temp_rect.height)) / float(net_height)) * seg_height) - rang_y;

        //std::cout << "rang:" << rang_x << " " << rang_y << " " << rang_w << " "  << rang_h << std::endl;
        
        rang_w = MAX(rang_w, 1);
        rang_h = MAX(rang_h, 1);
        if (rang_x + rang_w > seg_width) {
            if (seg_width - rang_x > 0)
                rang_w = seg_width - rang_x;
            else
                rang_x -= 1;
        }
        if (rang_y + rang_h > seg_height) {
            if (seg_height - rang_y > 0)
                rang_h = seg_height - rang_y;
            else
                rang_y -= 1;
        }
        //std::cout << "rang:" << rang_x << " " << rang_y << " " << rang_w << " "  << rang_h << std::endl;

        std::vector<cv::Range> roi_rangs;
        roi_rangs.push_back(cv::Range(0, 1));
        roi_rangs.push_back(cv::Range::all());
        roi_rangs.push_back(cv::Range(rang_y, rang_h + rang_y));
        roi_rangs.push_back(cv::Range(rang_x, rang_w + rang_x));
        //std::cout << roi_rangs << std::endl;
        //std::cout << "before crop" << std::endl;

        //crop
        cv::Mat temp_mask_protos = mask_protos(roi_rangs).clone();
        // std::cout << "out1" << std::endl;
        cv::Mat protos = temp_mask_protos.reshape(0, {seg_channels, rang_w * rang_h});
        //std::cout << "protos shape "<< protos.size << std::endl;
        //std::cout << "mask proposal shape " << maskProposal.size << std::endl;
        cv::Mat matmul_res = (maskProposal.t()*protos).t();
        //std::cout << "matmul_res shape " <<matmul_res.size<< std::endl;
        cv::Mat masks_feature = matmul_res.reshape(1, {rang_h, rang_w});
        //std::cout << "masks_feature shape " << masks_feature.size << std::endl;
        cv::Mat dest, mask;
        
        //sigmoid
        cv::exp(-masks_feature, dest);
        dest = 1.0 / (1.0 + dest);

        int left = floor(((net_width / float(seg_width)) * ((temp_rect.x / float(net_width)) * seg_width)));
        int top = floor(((net_height / float(seg_height)) * ((temp_rect.y / float(net_height)) * seg_height)));
        int width = ceil((net_width / float(seg_width)) * (((((temp_rect.x + temp_rect.width)) / float(net_width)) * seg_width) - rang_x));
        int height = ceil((net_height / float(seg_height)) * (((((temp_rect.y + temp_rect.height)) / float(net_height)) * seg_height) - rang_y));
        //std::cout << "mask:" << left << " " << top << " " << width << " "  << height << std::endl;
        //std::cout << width << " - " << height << std::endl;
        
        cv::resize(dest, mask, cv::Size(width, height), cv::INTER_NEAREST);
        //std::cout << mask.size << std::endl;
        //std::cout << temp_rect << std::endl;
        //std::cout << temp_rect - cv::Point(left, top) << std::endl;
        mask = mask(temp_rect - cv::Point(left, top)) > 0.5;
        //mask = mask(cv::Rect(temp_rect.tl() * params[0], temp_rect.br() * params[1]) - cv::Point(left + params[2], top + params[3])) > 0.5;
        
        //cv::resize(mask, mask, cv::Size(), params[0], params[1], cv::INTER_NEAREST);
        return mask;
    }
}