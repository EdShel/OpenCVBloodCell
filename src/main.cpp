#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml.hpp>
#include <iostream>

typedef cv::Point3_<uint8_t> byte3;
typedef float float1;

cv::Mat img;
double threshold = 0.30;
double minSize = 20;

cv::Mat createImpulseImage(cv::Mat source, double k)
{
    int channels = img.channels();
    CV_Assert(channels == 3);
    int w = img.cols;
    int h = img.rows;
    cv::Mat result(cv::Size(w, h), CV_32FC1);

    for (int y = 0; y < h; y++)
    {
        float1 *resultRowPointer = result.ptr<float1>(y);
        byte3 *sourceRowPointer = img.ptr<byte3>(y);

        double maxM = -999.0;
        for (int x = 0; x < w; x++)
        {
            byte3 pixel = sourceRowPointer[x];
            double m = (pixel.x + pixel.y + pixel.z) / 255.0 / 3.0;
            maxM = std::max(maxM, m);
        }

        double s = 0;
        for (int x = 0; x < w; x++)
        {
            byte3 pixel = sourceRowPointer[x];
            double m = (pixel.x + pixel.y + pixel.z) / 255.0 / 3.0;
            s += m;

            float1 b = 1;
            if (s >= k * maxM)
            {
                b = 0;
                s -= k * maxM;
            }

            resultRowPointer[x] = b;
        }
    }

    return result;
}

void detect()
{
    if (img.empty())
    {
        return;
    }

    double ks[] = {0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1};

    cv::Mat sum = cv::Mat::zeros(img.size(), CV_32FC1);
    for (int i = 0; i < sizeof(ks) / sizeof(double); i++)
    {
        cv::Mat impulse = createImpulseImage(img, ks[i]) / (sizeof(ks) / sizeof(double));
        sum += impulse;
    }

    cv::blur(sum, sum, cv::Size(7, 7));

    cv::Mat thr;
    cv::threshold(sum, thr, threshold, 1, 0);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    thr.convertTo(thr, CV_8U, 255);

    cv::findContours(thr, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat renderImage = img.clone();
    for (int i = 0; i < contours.size(); i++)
    {
        if (cv::contourArea(contours[i]) > minSize)
        {
            cv::Rect r = cv::boundingRect(cv::Mat(contours[i]));
            cv::rectangle(renderImage, r, cv::Scalar(0, 0, 255));
            cv::drawContours(renderImage, contours, (int)i, cv::Scalar(0, 255, 255), 1, cv::LINE_8, hierarchy, 0);
        }
    }

    cv::Mat resized;
    cv::resize(renderImage, resized, cv::Size(), 3.0, 3.0, cv::InterpolationFlags::INTER_NEAREST);
    cv::imshow("Result", resized);
}

int main(int argc, char *argv[])
{
    int init = 30;
    cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("Threshold", "Result", nullptr, 100, [](int value, void*) {
        threshold = value / 100.0;
        detect();
    });
    cv::createTrackbar("Min size", "Result", nullptr, 100000, [](int value, void*) {
        minSize = value / 10.0;
        detect();
    });

    std::vector<std::string> files;
    cv::glob("../img/*.jpg", files, false);
    for (int i = 0; i < files.size(); i++)
    {
        std::string imgPath = files[i];
        std::cout << imgPath << std::endl;
        img = cv::imread(imgPath);
        detect();

        int escape = 27;
        if (cv::waitKey(0) == escape)
        {
            return 0;
        }
    }

    return 0;
}