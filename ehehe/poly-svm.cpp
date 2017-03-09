#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2\imgproc.hpp>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\xfeatures2d.hpp>
#include <windows.h>
#define NTRAINING_SAMPLES   100         // Number of training samples per class
#define FRAC_LINEAR_SEP     0.9f        // Fraction of samples which compose the linear separable part

using namespace cv;
using namespace std;

void getLabels(ifstream &in, Mat &labels){
	int temp;
	for (int i = 0; !in.eof(); i++){
		in >> temp;
		Mat row = cv::Mat::ones(1, 1, CV_32SC1);  // 3 cols
		row.at<float>(0, 0) = temp;
		labels.push_back(row);
	}
}

void getTrainingData(ifstream &in, Mat &data){
	string line;
	double temp;
	while (getline(in, line)){
		Mat row = cv::Mat::ones(1, 128, CV_32FC1);
		std::string delimiter = ",";

		size_t pos = 0;
		int count = 0;
		std::string token;
		while ((pos = line.find(delimiter)) != std::string::npos) {
			token = line.substr(0, pos);
			line.erase(0, pos + delimiter.length());
			row.at<float>(0, count++) = stod(token);
		}
		row.at<float>(0, count++) = stod(line);
		data.push_back(row);
	}

}

void siftOperation(Mat neuron, vector<KeyPoint> &keypoints, Mat &descriptors){
	cvtColor(neuron, neuron, CV_BGR2GRAY);
	Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
	f2d->detect(neuron, keypoints);
	f2d->compute(neuron, keypoints, descriptors);
}


void readAndWrite(string file, Ptr<ml::SVM> &svm){
	Mat img;
	img = imread(file, CV_LOAD_IMAGE_COLOR);

	if (!img.data){
		cout << "Could not find or open the neuron1 image." << endl;
		return;
	}
	vector<KeyPoint> keypoints;
	Mat descriptors;
	siftOperation(img, keypoints, descriptors);

	//------------------------ 4. Show the decision regions ----------------------------------------
	Vec3b blue(100, 0, 0), red(0, 0, 100);
	int thick = -1, lineType = 8;
	for (int i = 0; i < descriptors.rows; i++){
		float response = svm->predict(descriptors.row(i));
		float x = keypoints[i].pt.x;
		float y = keypoints[i].pt.y;
		if (response >= 1)    //img.at<Vec3b>(y, x) = red;
			circle(img, Point((int)x, (int)y), 3, red, thick, lineType);
		else if (response <1)    //img.at<Vec3b>(y, x) = blue;
			circle(img, Point((int)x, (int)y), 3, blue, thick, lineType);
	}

	//imshow("Keypoints 1", img);
	imwrite(file.substr(0, file.length() - 4) + "+tested.png", img);

}

//System::Void button1_Click(System::Object^ sender, System::EventArgs^ e){
//	test a;
//	SaveFileDialog ^ saveFileDialog1 = gcnew SaveFileDialog();
//	saveFileDialog->Filter = "Text file (*.txt)|*.txt";
//	saveFileDialog->Title = "Save a text file";
//	//CODE REQUIRED TO SAVE s2 AS .TXT FILE
//	saveFileDialog1->ShowDialog();
//}

int main()
{
	const int WIDTH = 512, HEIGHT = 512;
	Mat I = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);

	//--------------------- 1. Set up training data randomly ---------------------------------------
	Mat trainData(0, 128, CV_32FC1);
	Mat labels(0, 1, CV_32SC1);

	
	ifstream in;
	string labelFile = "labels_training.txt";
	string dataFile = "descriptors_training.txt";

	in.open(labelFile.c_str());
	if (in.fail()){
		cout << "The label file could not be opened.\n";
	}
	else{
		getLabels(in, labels);
		in.clear();
		in.close();
		in.open(dataFile.c_str());
		if (in.fail()){
			cout << "The descriptor file could not be read.\n";
		}
		else{
			getTrainingData(in, trainData);
		}
	}
	

	//------------------------ 2. Set up the support vector machines parameters --------------------
	Ptr<ml::SVM> svm = Algorithm::load<ml::SVM>("classifier.xml");
	svm->setType(ml::SVM::C_SVC);
	svm->setC(0.1);
	svm->setKernel(ml::SVM::POLY);
	svm->setDegree(1);
	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER, (int)1e7, 1e-6));

	//------------------------ 3. Train the svm ----------------------------------------------------
	cout << "Starting training process" << endl;
	//svm->train(trainData, ml::SampleTypes::ROW_SAMPLE, labels);
	//svm->save("classifier.xml");
	// svm->load("classifier.xml");
	cout << "Finished training process" << endl;


	string file = "image_10.png";
	readAndWrite(file, svm);
	file = "image_1.png";
	readAndWrite(file, svm);
	file = "image_2.png";
	readAndWrite(file, svm);
	file = "image_3.png";
	readAndWrite(file, svm);
	file = "image_4.png";
	readAndWrite(file, svm);
	file = "image_5.png";
	readAndWrite(file, svm);
	file = "image_6.png";
	readAndWrite(file, svm);
	file = "image_7.png";
	readAndWrite(file, svm);
	file = "image_8.png";
	readAndWrite(file, svm);
	file = "image_9.png";
	readAndWrite(file, svm);
	file = "image_11.png";
	readAndWrite(file, svm);


	cout << "Finished.\n";
	waitKey(0);
}