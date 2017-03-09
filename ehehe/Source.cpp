//#include <opencv2/opencv.hpp>
//#include <opencv2\core\core.hpp>
//#include <opencv2\highgui\highgui.hpp>
//#include <opencv2\features2d\features2d.hpp>
//#include <opencv2\xfeatures2d.hpp>
//
//
//using namespace std;
//using namespace cv;
//
//int main(){
//	Mat neuron1, neuron2, neuron3;
//	string file1, file2, file3;
//	file1 = "ZSeries-neuron1AFPansiso1-182";
//	file2 = "ZSeries-Neuron1baseline3-021";
//	file3 = "ZSeries-NeuronsBAS2-120";
//	neuron1 = imread(file1 + ".png", CV_LOAD_IMAGE_COLOR);
//	neuron2 = imread(file2 + ".png", CV_LOAD_IMAGE_COLOR);
//	neuron3 = imread(file3 + ".png", CV_LOAD_IMAGE_COLOR);
//	if (!neuron1.data){
//		cout << "Could not find or open the neuron1 image." << endl;
//		return -1;
//	}
//	if (!neuron2.data){
//		cout << "Could not find or open the neuron2 image." << endl;
//		return -1;
//	}
//	if (!neuron3.data){
//		cout << "Could not find or open the neuron3 image." << endl;
//		return -1;
//	}
//
//	cvtColor(neuron1, neuron1, CV_BGR2GRAY);
//	cvtColor(neuron2, neuron2, CV_BGR2GRAY);
//	cvtColor(neuron3, neuron3, CV_BGR2GRAY);
//
//	Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
//	vector<KeyPoint> keypoints1, keypoints2, keypoints3;
//	f2d->detect(neuron1, keypoints1);
//	f2d->detect(neuron2, keypoints2);
//	f2d->detect(neuron3, keypoints3);
//
//	Mat descriptors1, descriptors2, descriptors3;
//	f2d->compute(neuron1, keypoints1, descriptors1);
//	f2d->compute(neuron2, keypoints2, descriptors2);
//	f2d->compute(neuron3, keypoints3, descriptors3);
//
//	drawKeypoints(neuron1, keypoints1, neuron1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
//	drawKeypoints(neuron2, keypoints2, neuron2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
//	drawKeypoints(neuron3, keypoints3, neuron3, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
//
//	imwrite(file1 + "_sift.png", neuron1);
//	imwrite(file2 + "_sift.png", neuron2);
//	imwrite(file3 + "_sift.png", neuron3);
//
//	imshow("Keypoints 1", neuron1);
//	imshow("Keypoints 2", neuron2);
//	imshow("Keypoints 3", neuron3);
//	waitKey(0);
//	/*Mat image, image2;*/
//	//image = imread("a.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
//	//image2 = imread("b.jpg", CV_LOAD_IMAGE_COLOR);
//	//cvtColor(image, image, CV_BGR2GRAY);
//	//cvtColor(image2, image2, CV_BGR2GRAY);
//
//	//if (!image.data)                              // Check for invalid input
//	//{
//	//	cout << "Could not open or find the image" << std::endl;
//	//	return -1;
//	//}
//
//	//namedWindow("Display window", WINDOW_AUTOSIZE);// Create a window for display.
//	//Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
//	//vector<KeyPoint> keypoints1, keypoints2;
//	//f2d->detect(image, keypoints1);
//	//f2d->detect(image2, keypoints2);
//	//Mat descriptors1, descriptors2;
//	//f2d->compute(image, keypoints1, descriptors1);
//	//f2d->compute(image2, keypoints2, descriptors2);
//	//BFMatcher matcher;;
//	//vector< DMatch > matches;
//	//matcher.match(descriptors1, descriptors2, matches);
//	//Mat matchesimg;
//
//	//drawKeypoints(image, keypoints1, image, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
//	//drawKeypoints(image2, keypoints2, image2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
//	//drawMatches(image, keypoints1, image2, keypoints2, matches, matchesimg, Scalar::all(-1));
//	//
//	////SiftFeatureDetector detector;
//	//imshow("Keypoints 1", image);
//	//imshow("Keypoints 2", image2);
//	//imshow("matchesimg ", matchesimg);
//
//	//waitKey(0);  
//	//// Wait for a keystroke in the window
//	//FileStorage fs("test.xml", FileStorage::WRITE);
//	//write(fs, "keypoints1", keypoints1);
//	//write(fs, "keypoints2", keypoints2);
//	//write(fs, "descriptors1", descriptors1);
//	//write(fs, "descriptors2", descriptors2);
//
//
//	///*fs << "descriptors1 " << descriptors1 ;
//	//fs << "descriptors2 " << descriptors2 ;*/
//	//fs.release();
//
//	return 0;
//	//VideoCapture cap(0);
//	//Mat new_frame;
//	//while (true){
//	//	cap >> new_frame;
//	//	if (new_frame.empty()){
//	//		break;
//	//	}
//	//	imshow("camera", new_frame);
//	//	// press ESC to quit software
//	//	if ((int)waitKey(10) == 27){
//	//		break;
//	//	}
//	//}
//}
//
//
////#include <iostream>
////
////#include "opencv2/opencv_modules.hpp"
////#define HAVE_OPENCV_XFEATURES2D
////
////#ifdef HAVE_OPENCV_XFEATURES2D
////
////#include <opencv2/features2d.hpp>
////#include <opencv2/xfeatures2d.hpp>
////#include <opencv2/imgcodecs.hpp>
////#include <opencv2/opencv.hpp>
////#include <vector>
////
////// If you find this code useful, please add a reference to the following paper in your work:
////// Gil Levi and Tal Hassner, "LATCH: Learned Arrangements of Three Patch Codes", arXiv preprint arXiv:1501.03719, 15 Jan. 2015
////
////using namespace std;
////using namespace cv;
////
////const float inlier_threshold = 2.5f; // Distance threshold to identify inliers
////const float nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio
////
////int main(void)
////{
////	Mat img1 = imread("../data/graf1.png", IMREAD_GRAYSCALE);
////	Mat img2 = imread("../data/graf3.png", IMREAD_GRAYSCALE);
////
////
////	Mat homography;
////	FileStorage fs("../data/H1to3p.xml", FileStorage::READ);
////
////	fs.getFirstTopLevelNode() >> homography;
////
////	vector<KeyPoint> kpts1, kpts2;
////	Mat desc1, desc2;
////
////	Ptr<cv::ORB> orb_detector = cv::ORB::create(10000);
////
////	Ptr<xfeatures2d::LATCH> latch = xfeatures2d::LATCH::create();
////
////
////	orb_detector->detect(img1, kpts1);
////	latch->compute(img1, kpts1, desc1);
////
////	orb_detector->detect(img2, kpts2);
////	latch->compute(img2, kpts2, desc2);
////
////	BFMatcher matcher(NORM_HAMMING);
////	vector< vector<DMatch> > nn_matches;
////	matcher.knnMatch(desc1, desc2, nn_matches, 2);
////
////	vector<KeyPoint> matched1, matched2, inliers1, inliers2;
////	vector<DMatch> good_matches;
////	for (size_t i = 0; i < nn_matches.size(); i++) {
////		DMatch first = nn_matches[i][0];
////		float dist1 = nn_matches[i][0].distance;
////		float dist2 = nn_matches[i][1].distance;
////
////		if (dist1 < nn_match_ratio * dist2) {
////			matched1.push_back(kpts1[first.queryIdx]);
////			matched2.push_back(kpts2[first.trainIdx]);
////		}
////	}
////
////	for (unsigned i = 0; i < matched1.size(); i++) {
////		Mat col = Mat::ones(3, 1, CV_64F);
////		col.at<double>(0) = matched1[i].pt.x;
////		col.at<double>(1) = matched1[i].pt.y;
////
////		col = homography * col;
////		col /= col.at<double>(2);
////		double dist = sqrt(pow(col.at<double>(0) - matched2[i].pt.x, 2) +
////			pow(col.at<double>(1) - matched2[i].pt.y, 2));
////
////		if (dist < inlier_threshold) {
////			int new_i = static_cast<int>(inliers1.size());
////			inliers1.push_back(matched1[i]);
////			inliers2.push_back(matched2[i]);
////			good_matches.push_back(DMatch(new_i, new_i, 0));
////		}
////	}
////
////	Mat res;
////	drawMatches(img1, inliers1, img2, inliers2, good_matches, res);
////	imwrite("../../samples/data/latch_res.png", res);
////
////
////	double inlier_ratio = inliers1.size() * 1.0 / matched1.size();
////	cout << "LATCH Matching Results" << endl;
////	cout << "*******************************" << endl;
////	cout << "# Keypoints 1:                        \t" << kpts1.size() << endl;
////	cout << "# Keypoints 2:                        \t" << kpts2.size() << endl;
////	cout << "# Matches:                            \t" << matched1.size() << endl;
////	cout << "# Inliers:                            \t" << inliers1.size() << endl;
////	cout << "# Inliers Ratio:                      \t" << inlier_ratio << endl;
////	cout << endl;
////	return 0;
////}
////
////#else
////
////int main()
////{
////	std::cerr << "OpenCV was built without xfeatures2d module" << std::endl;
////	return 0;
////}
////
////#endif
//
//
//
//
