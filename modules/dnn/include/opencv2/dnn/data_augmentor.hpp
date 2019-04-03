#include <string>
#include <functional>
#include <random>
#include <vector>

#include <opencv2/core.hpp>


#ifndef OPENCV_DATAAUGMENTOR
#define OPENCV_DATAAUGMENTOR

namespace cv { namespace dnn {

class Method
{
public:
    Method(std::function<cv::Mat(const cv::Mat&)>& augmentor);

    double getProbability() const;

    cv::Mat runAugmentor(const cv::Mat& image);
private:
    std::function<cv::Mat(const cv::Mat&)>& augmentor;

    double probability;
};

class DataAugment
{
public:
    DataAugment(const std::vector<cv::Mat>& images);

    ~DataAugment();

    void addMethod(std::function<cv::Mat(const cv::Mat&)>& augmentor);

    void createSample(std::vector<cv::Mat>& resImages);
private:
    std::string outputDirectory;

    std::vector<Method> methods;

    const std::vector<cv::Mat> images;

    static cv::Mat flipAugmentor(const cv::Mat& image);
};

}
}

#endif