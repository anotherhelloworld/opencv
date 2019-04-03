#include "opencv2/dnn/data_augmentor.hpp"

namespace cv { namespace dnn {

DataAugment::DataAugment(const std::vector<cv::Mat>& images)
    : images(images)
{
    std::function<cv::Mat(const cv::Mat&)> flipFunc = DataAugment::flipAugmentor;
    methods.push_back(Method(flipFunc));
}

void DataAugment::createSample(std::vector<cv::Mat>& resImages)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> imageDis(0, images.size() - 1);
    std::uniform_real_distribution<> probabilityDis(0.0, 1.0);

    cv::Mat image = images[imageDis(gen)];

    for (auto& method: methods)
    {
        double r = probabilityDis(gen);
        if (r <= method.getProbability())
        {
            resImages.push_back(method.runAugmentor(image));
        }
    }

}

void DataAugment::addMethod(std::function<cv::Mat(const cv::Mat&)>& augmentor)
{
    methods.push_back(Method(augmentor));
}

cv::Mat DataAugment::flipAugmentor(const cv::Mat& image)
{
    cv::Mat flipped;
    cv::flip(image, flipped, 1);
    return flipped;
}

Method::Method(std::function<cv::Mat(const cv::Mat&)>& augmentor)
    : augmentor(augmentor)
{
}

double Method::getProbability() const
{
    return probability;
}

cv::Mat Method::runAugmentor(const cv::Mat& image)
{
    return augmentor(image);
}

}
}