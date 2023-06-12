#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml.hpp>
#include <iostream>

typedef cv::Point3_<uint8_t> byte3;
typedef float float1;

cv::Mat img;

int main(int argc, char *argv[])
{
    std::string imgPath = "../img/k_001.jpg";
    img = cv::imread(imgPath);

    cv::TrackbarCallback updateImage = [](int value, void *)
    {
        double k = value / 1000.0;

        int channels = img.channels();
        CV_Assert(channels == 3);
        int w = img.cols;
        int h = img.rows;
        cv::Mat res(cv::Size(w, h), CV_32FC1);

        for (int y = 0; y < h; y++)
        {
            float1 *resultRowPointer = res.ptr<float1>(y);
            byte3 *rowPointer = img.ptr<byte3>(y);

            double maxM = -999.0;
            for (int x = 0; x < w; x++)
            {
                byte3 pixel = rowPointer[x];
                double m = (pixel.x + pixel.y + pixel.z) / 255.0 / 3.0;
                maxM = std::max(maxM, m);
            }

            double s = 0;
            for (int x = 0; x < w; x++)
            {
                byte3 pixel = rowPointer[x];
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

        cv::Mat resized;
        cv::resize(res, resized, cv::Size(), 3.0, 3.0, cv::InterpolationFlags::INTER_NEAREST);
        cv::imshow("Result", resized);
    };
    int init = 100;
    cv::namedWindow("Result", cv::WINDOW_AUTOSIZE);
    cv::createTrackbar("k", "Result", &init, 10000, updateImage);


    cv::waitKey(0);

    return 0;
}