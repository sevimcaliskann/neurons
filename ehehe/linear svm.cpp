//#include <opencv2/core/core.hpp>
//#include <opencv2\imgproc.hpp>
//#include <opencv2/highgui/highgui.hpp>
////#include "opencv2\imgcodecs.hpp"
//#include <opencv2/ml/ml.hpp>
//
//using namespace cv;
//using namespace cv::ml;
//
//int main(int, char**){
//
//	//Data for visual representation
//	int width = 512, height = 512;
//	Mat image = Mat::zeros(height, width, CV_8UC3);
//
//	//Set up training data
//	float labels[4] = { 1.0, -1.0, -1.0, -1.0 };
//	Mat labelsMat(4, 1, CV_32SC1, labels);
//
//	//Set up SVM's parameters
//	float trainingData[4][2] = { { 501.0, 10.0 }, { 255.0, 10.0 }, { 501.0, 255.0 }, { 10.0, 501.0 } };
//	Mat trainingDataMat(4, 2, CV_32FC1, trainingData);
//
//	Ptr<ml::SVM> svm = ml::SVM::create();
//	svm->setType(ml::SVM::C_SVC);
//	svm->setKernel(ml::SVM::POLY);
//	svm->setDegree(1);
//	svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 100, 1e-6));
//	svm->train(trainingDataMat, ml::SampleTypes::ROW_SAMPLE, labelsMat);
//
//	Vec3b green(0, 255, 0), blue(255, 0, 0);
//	for (int i = 0; i < image.rows; ++i){
//		for (int j = 0; j < image.cols; ++j){
//			Mat sampleMat = (Mat_<float>(1, 2) << j, i);
//			float response = (float)svm->predict(sampleMat);
//			if (response == 1)
//				image.at<Vec3b>(i, j) = green;
//			else if (response == -1)
//				image.at<Vec3b>(i, j) = blue;
//
//		}
//	}
//
//	int thickness = -1;
//	int lineType = 8;
//	circle(image, Point(501, 10), 5, Scalar(0, 0, 0), thickness, lineType);
//	circle(image, Point(255, 10), 5, Scalar(255, 255, 255), thickness, lineType);
//	circle(image, Point(501, 255), 5, Scalar(255, 255, 255), thickness, lineType);
//	circle(image, Point(10, 501), 5, Scalar(255, 255, 255), thickness, lineType);
//
//	thickness = 2;
//	lineType = 8;
//	Mat sv = svm->getSupportVectors();
//	for (int i = 0; i < sv.rows; i++){
//		const float *v = sv.ptr<float>(i);
//		circle(image, Point((int)v[0], (int)v[1]), 6, Scalar(128, 128, 128), thickness, lineType);
//	}
//
//	imwrite("result.png", image);
//	imshow("SVM Simple Example", image);
//	waitKey(0);
//
//
//}