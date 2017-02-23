#include "stdafx.h"

#include <iostream>
#include <fstream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "tserial.h"
#include "usb_control.h"
#include <math.h>
//#include <stdio.h>
//#include <conio.h>
//#include <atlstr.h>
//#include <afx.h>

using namespace cv;
using namespace std;

serial com;

//------------------------------Matlab Communication File Streams------------------------------

ifstream start_file;
ofstream mode_file;
ofstream exit_file;
ofstream flagfile;
ofstream opfile;
ofstream number_egfile;

//---------------------------------------------------------------------------------------------

//------------------------------Variables for Multi Robot Control------------------------------


int mode = 0;
int calib_flag = 0, fing_calib_flag = 0, count_flag = 0, id_flag = 0, exit_flag = 0;
char m;
int controlled_bot_id;

unsigned char id;
int n_bots;
int x_bots[100] = { 0 }, y_bots[100] = { 0 };
int x_bots_temp[100] = { 0 }, y_bots_temp[100] = { 0 };
int x_bots_old[100] = { 0 }, y_bots_old[100] = { 0 };

int phi_bots[100] = { 0 }, phi_zoom[100] = { 0 };
int e_phi_sum[100] = { 0 }, e_x_sum[100] = { 0 }, e_y_sum[100] = { 0 };
int e_phi = 0, e_x_bots = 0, e_y_bots = 0, e_obst = 0, count_ob;
int phi_g, phi_obst;
int vel;
int l = 0, r = 0;
unsigned char motion;

double bot_obst_dist[100][100] = { 0 };

int obst_flag[100] = { 0 };
int reached = 0;

int fing_count = 0, fing_dist = 0, out_flag, in_flag;;

Point locbotg[100] = { 0 }, locbotg_old[100] = { 0 }, locbotg_temp[100] = { 0 };
Point locr[100] = { 0 }, locr_old[100] = { 0 }, locr_temp[100] = { 0 };
Point locb[100] = { 0 }, locb_old[100] = { 0 }, locb_temp[100] = { 0 };
Point locfing[100] = { 0 }, locfing_temp[100] = { 0 };
//Point goals[100] = { 0 };
Point obstacles[100] = { 0 };

Point triangle_locs [5] = { Point(60, 200), Point(110, 120), Point(160, 60), Point(210, 120), Point(260, 200) };
Point v_locs[5] = { Point(60, 40), Point(110, 120), Point(160, 200), Point(210, 120), Point(260, 40) };
Point square_locs[5] = { Point(160, 120), Point(90, 50), Point(90, 190), Point(230, 190), Point(230, 50) };
Point rectangle_locs[5] = { Point(160, 120), Point(50, 50), Point(50, 190), Point(270, 190), Point(270, 50) };
Point circle_locs[5] = { Point(160, 40), Point(83, 95), Point(112, 184), Point(208, 184), Point(237, 95) };
Point t_locs[5] = { Point(160, 60), Point(60, 60), Point(160, 130), Point(160, 200), Point(260, 60) };
Point pentagon_locs[5] = { Point(160, 40), Point(83, 95), Point(112, 184), Point(208, 184), Point(237, 95) };
Point inverted_v_locs[5] = { Point(60, 200), Point(110, 120), Point(160, 40), Point(210, 120), Point(260, 200) };
Point diamond_locs[5] = { Point(160, 120), Point(160, 40), Point(80, 120), Point(160, 200), Point(240, 120) };
Point line_locs[5] = { Point(32, 120), Point(96, 120), Point(160, 120), Point(224, 120), Point(288, 120) };

int pred;

Point shapes[100][5] = { 0 };
char shape_names[100][20];

Point goal_temp[5] = { 0 };


Point goals[100] = { 0 };
unsigned char taken[100] = { 0 };

Point side_goals[100] = { 0 };


Mat database[100] = { Mat::zeros(Size(320, 240), CV_8UC3) };

void init_database()
{
	database[0] = imread("Disp-V.jpeg", CV_LOAD_IMAGE_UNCHANGED);
	database[1] = imread("Disp-Rectangle.jpeg", CV_LOAD_IMAGE_UNCHANGED);
	database[2] = imread("Disp-Circle.jpeg", CV_LOAD_IMAGE_UNCHANGED);
	database[3] = imread("Disp-T.jpeg", CV_LOAD_IMAGE_UNCHANGED);
	database[4] = imread("Disp-Inverted V.jpeg", CV_LOAD_IMAGE_UNCHANGED);
	database[5] = imread("Disp-Line.jpeg", CV_LOAD_IMAGE_UNCHANGED);
	//database[6] = imread("DB-Pentagon.jpeg", CV_LOAD_IMAGE_GRAYSCALE);
	//database[7] = imread("DB-Inverted V.jpeg", CV_LOAD_IMAGE_GRAYSCALE);
	//database[8] = imread("DB-Diamond.jpeg", CV_LOAD_IMAGE_GRAYSCALE);
	//database[9] = imread("DB-Line.jpeg", CV_LOAD_IMAGE_GRAYSCALE);
}


vector<vector<Point>> contours_green;
vector<vector<Point>> contours_red;
vector<vector<Point>> contours_blue;
vector<vector<Point>> all_col;

vector<Point> path;


//---------------------------------------------------------------------------------------------


//------------------------------Variables for Color Detection Thresholds Values----------------

int maxHG = 152, minHG = 99, maxSG = 253, minSG = 65, maxVG = 255, minVG = 101;
int minHR = 138, minSR = 71, minVR = 101, maxHR = 255, maxSR = 255, maxVR = 255;
int minHB = 81, minSB = 51, minVB = 49, maxHB = 152, maxSB = 255, maxVB = 255;
int minHFing = 85, minSFing = 51, minVFing = 49, maxHFing = 152, maxSFing = 255, maxVFing = 255;
int minHAll = 85, minSAll = 43, minVAll = 99, maxHAll = 255, maxSAll = 158, maxVAll = 255;

//---------------------------------------------------------------------------------------------

//-----------------------------Variables for Gesture Recognition-------------------------------

Point old_c = Point(-1, -1);
Point new_c = Point(-1, -1);
Point cold_c = Point(-1, -1);
Point cnew_c = Point(-1, -1);

int size = 50000;

//---------------------------------------------------------------------------------------------

//------------------------------Camera Objects and Matrices for Images-------------------------

VideoCapture webcam(1);
VideoCapture cam(2);

HWND hDesktopWnd = GetDesktopWindow();

int coco = 0;


Mat img;
Mat img2;
Mat img_hsv;

Mat thresh_blue = Mat::zeros(img.size(), CV_8UC1);
Mat thresh_yellow = Mat::zeros(img.size(), CV_8UC1);
Mat thresh_red = Mat::zeros(img.size(), CV_8UC1);
Mat thresh_finger = Mat::zeros(img.size(), CV_8UC1);
Mat blank = Mat::zeros(Size(320, 240), CV_8UC3);
Mat blobg = Mat::zeros(img.size(), CV_8UC1);
Mat blobr = Mat::zeros(img.size(), CV_8UC1);
Mat blobb = Mat::zeros(img.size(), CV_8UC1);

Mat drawing1 = Mat::zeros(img.size(), CV_8UC3);
Mat drawing2 = Mat::zeros(img.size(), CV_8UC3);
Mat drawing3 = Mat::zeros(img.size(), CV_8UC3);
Mat drawing4 = Mat::zeros(img.size(), CV_8UC3);


Mat thresh2;
Mat gray;

//---------------------------------------------------------------------------------------------

//--------------------------Camera and System Calibrations-------------------------------------

void id_bot_location();
void calibrate();
void calibrate_gesture_markers();


//---------------------------------------------------------------------------------------------

//--------------------------Multi-Robot Control Functions List---------------------------------

void locate_green_temp_quick();
void calib_id_quick();
void locate_bots_quick();
void get_angles();
int angle_in_range(int phi);
void go_to_goals();
void avoid_obstacles_ind(int id);
double point_distance(Point p1, Point p2); 
void go_to_goal_ind(int id);

//---------------------------------------------------------------------------------------------

//-------------------------Gesture Recognition Function List-----------------------------------

void draw_and_save(int num);
void recognise_gesture();


//---------------------------------------------------------------------------------------------

void init_shapes()
{
	/*Point shapes[6][5] = { { Point(60, 40), Point(110, 120), Point(160, 200), Point(210, 120), Point(260, 40) },		//V
	{ Point(270, 50), Point(50, 50), Point(50, 190), Point(270, 190), Point(160, 120) },		//Rectangle
	{ Point(160, 40), Point(83, 95), Point(112, 184), Point(208, 184), Point(237, 95) },		//Circle
	{ Point(160, 60), Point(60, 60), Point(160, 130), Point(160, 200), Point(260, 60) },		//T
	{ Point(60, 200), Point(110, 120), Point(160, 40), Point(210, 120), Point(260, 200) },		//Inverted V
	{ Point(32, 24), Point(96, 72), Point(160, 120), Point(224, 168), Point(288, 216) } };		// '\' Line*/
	shapes[0][0] = Point(60, 40);
	shapes[0][1] = Point(110, 120);
	shapes[0][2] = Point(160, 200);
	shapes[0][3] = Point(210, 120);
	shapes[0][4] = Point(260, 40);
	shapes[1][0] = Point(270, 50);
	shapes[1][1] = Point(50, 50);
	shapes[1][2] = Point(50, 190);
	shapes[1][3] = Point(270, 190);
	shapes[1][4] = Point(160, 120);
	shapes[2][0] = Point(160, 40);
	shapes[2][1] = Point(83, 95);
	shapes[2][2] = Point(112, 184);
	shapes[2][3] = Point(208, 184);
	shapes[2][4] = Point(237, 95);
	shapes[3][0] = Point(160, 60);
	shapes[3][1] = Point(60, 60);
	shapes[3][2] = Point(160, 130);
	shapes[3][3] = Point(160, 200);
	shapes[3][4] = Point(260, 60);
	shapes[4][0] = Point(60, 200);
	shapes[4][1] = Point(110, 120);
	shapes[4][2] = Point(160, 40);
	shapes[4][3] = Point(210, 120);
	shapes[4][4] = Point(260, 200);
	shapes[5][0] = Point(32, 24);
	shapes[5][1] = Point(96, 72);
	shapes[5][2] = Point(160, 120);
	shapes[5][3] = Point(224, 168);
	shapes[5][4] = Point(288, 216);
}

void init_shape_names()
{
	_snprintf_s(shape_names[0], 20, "V");
	_snprintf_s(shape_names[1], 20, "Rectangle");
	_snprintf_s(shape_names[2], 20, "Circle");
	_snprintf_s(shape_names[3], 20, "T");
	_snprintf_s(shape_names[4], 20, "Inverted V");
	_snprintf_s(shape_names[5], 20, " \'\\' Line");
}

void init_camera_quad()
{
	cam.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	cam.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	cam.set(CV_CAP_PROP_FPS, 20);
}

void init_webcam()
{
	webcam.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	webcam.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	webcam.set(CV_CAP_PROP_FPS, 30);
}

void init_bots_state()
{
	for (int i = 0; i < n_bots; i++)
	{
		com.send_data(0);
		Sleep(5);
	}
}

void clear_screen(char fill = ' ') 
{
	COORD tl = { 0, 0 };
	CONSOLE_SCREEN_BUFFER_INFO s;
	HANDLE console = GetStdHandle(STD_OUTPUT_HANDLE);
	GetConsoleScreenBufferInfo(console, &s);
	DWORD written, cells = s.dwSize.X * s.dwSize.Y;
	FillConsoleOutputCharacter(console, fill, cells, tl, &written);
	FillConsoleOutputAttribute(console, s.wAttributes, cells, tl, &written);
	SetConsoleCursorPosition(console, tl);
}

int angle_in_range(int phi)
{
	if (phi > 180)
	{
		return phi - 360;
	}
	else if (phi < -180)
	{
		return phi + 360;
	}
}

void calibrate()
{
	cam.read(img);


	//namedWindow("Blue Tracking", CV_WINDOW_AUTOSIZE);
	namedWindow("Blue Tracking", CV_WINDOW_AUTOSIZE);
	namedWindow("Red Tracking", CV_WINDOW_AUTOSIZE);
	namedWindow("Blue Blob", CV_WINDOW_AUTOSIZE);
	//namedWindow("Yellow Blob", CV_WINDOW_AUTOSIZE);
	namedWindow("Red Blob", CV_WINDOW_AUTOSIZE);
	//namedWindow("Blue Contour", CV_WINDOW_AUTOSIZE);
	//namedWindow("Yellow Contour", CV_WINDOW_AUTOSIZE);
	//namedWindow("Red Contour", CV_WINDOW_AUTOSIZE);
	namedWindow("Stream", CV_WINDOW_AUTOSIZE);

	//createTrackbar("maxH", "Blue Tracking", &maxHB, 255);
	//createTrackbar("minH", "Blue Tracking", &minHB, 255);
	//createTrackbar("maxS", "Blue Tracking", &maxSB, 255);
	//createTrackbar("minS", "Blue Tracking", &minSB, 255);
	//createTrackbar("maxV", "Blue Tracking", &maxVB, 255);
	//createTrackbar("minV", "Blue Tracking", &minVB, 255);
	createTrackbar("maxH", "Blue Tracking", &maxHG, 255);
	createTrackbar("minH", "Blue Tracking", &minHG, 255);
	createTrackbar("maxS", "Blue Tracking", &maxSG, 255);
	createTrackbar("minS", "Blue Tracking", &minSG, 255);
	createTrackbar("maxV", "Blue Tracking", &maxVG, 255);
	createTrackbar("minV", "Blue Tracking", &minVG, 255);
	createTrackbar("maxH", "Red Tracking", &maxHR, 255);
	createTrackbar("minH", "Red Tracking", &minHR, 255);
	createTrackbar("maxS", "Red Tracking", &maxSR, 255);
	createTrackbar("minS", "Red Tracking", &minSR, 255);
	createTrackbar("maxV", "Red Tracking", &maxVR, 255);
	createTrackbar("minV", "Red Tracking", &minVR, 255);


	while (1)
	{
		cam.read(img);
		cvtColor(img, img_hsv, CV_BGR2HSV);
		inRange(img_hsv, Scalar(minHG, minSG, minVG), Scalar(maxHG, maxSG, maxVG), thresh_yellow);
		inRange(img_hsv, Scalar(minHR, minSR, minVR), Scalar(maxHR, maxSR, maxVR), thresh_red);
		/*inRange(img_hsv, Scalar(minHB, minSB, minVB), Scalar(maxHB, maxSB, maxVB), thresh_blue);
		erode(thresh_blue, thresh_blue, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		dilate(thresh_blue, thresh_blue, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		dilate(thresh_blue, thresh_blue, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		erode(thresh_blue, thresh_blue, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));*/
		erode(thresh_red, thresh_red, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		dilate(thresh_red, thresh_red, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		dilate(thresh_red, thresh_red, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		erode(thresh_red, thresh_red, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		erode(thresh_yellow, thresh_yellow, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		dilate(thresh_yellow, thresh_yellow, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		dilate(thresh_yellow, thresh_yellow, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		erode(thresh_yellow, thresh_yellow, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		//imshow("Blue Blob", thresh_blue);
		imshow("Blue Blob", thresh_yellow);
		imshow("Red Blob", thresh_red);
		imshow("Stream", img);
		/*Moments asd = moments(thresh_yellow);
		cout << (asd.m00/255) << endl;*/
		if (waitKey(30) == 32)
		{
			break;
		}

	}
	destroyWindow("Blue Tracking");
	destroyWindow("Yellow Tracking");
	destroyWindow("Red Tracking");
	destroyWindow("Blue Blob");
	//destroyWindow("Yellow Blob");
	destroyWindow("Red Blob");
	//destroyWindow("Blue Contour");
	//destroyWindow("Yellow Contour");
	//destroyWindow("Red Contour");
	//destroyWindow("Stream");
	
}

void calibrate_gesture_markers()
{
	webcam.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	webcam.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	webcam.set(CV_CAP_PROP_FPS, 30);
	namedWindow("Finger Tracking", CV_WINDOW_NORMAL);
	namedWindow("Finger Blob", CV_WINDOW_AUTOSIZE);
	namedWindow("Webcam Stream", CV_WINDOW_AUTOSIZE);

	createTrackbar("maxH", "Finger Tracking", &maxHFing, 255);
	createTrackbar("minH", "Finger Tracking", &minHFing, 255);
	createTrackbar("maxS", "Finger Tracking", &maxSFing, 255);
	createTrackbar("minS", "Finger Tracking", &minSFing, 255);
	createTrackbar("maxV", "Finger Tracking", &maxVFing, 255);
	createTrackbar("minV", "Finger Tracking", &minVFing, 255);
	createTrackbar("Size", "Finger Tracking", &size, 200000);

	while (1)
	{
		webcam.read(img);
		flip(img, img, 1);
		cvtColor(img, img_hsv, CV_BGR2HSV);
		inRange(img_hsv, Scalar(minHFing, minSFing, minVFing), Scalar(maxHFing, maxSFing, maxVFing), thresh_finger);
		//inRange(img_hsv, Scalar(minHR, minSR, minVR), Scalar(maxHR, maxSR, maxVR), thresh_red);
		//inRange(img_hsv, Scalar(minHB, minSB, minVB), Scalar(maxHB, maxSB, maxVB), thresh_blue);
		/*erode(thresh_blue, thresh_blue, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		dilate(thresh_blue, thresh_blue, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		dilate(thresh_blue, thresh_blue, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		erode(thresh_blue, thresh_blue, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		erode(thresh_red, thresh_red, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		dilate(thresh_red, thresh_red, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		dilate(thresh_red, thresh_red, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		erode(thresh_red, thresh_red, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));*/
		erode(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		dilate(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		//imshow("Blue Blob", thresh_blue);
		imshow("Finger Blob", thresh_finger);
		//imshow("Red Blob", thresh_red);
		imshow("Webcam Stream", img);
		if (waitKey(30) == 32)
		{
			break;
		}
	}
}

void calibrate_all_finger_markers()
{
	destroyWindow("Finger Tracking");
	namedWindow("Finger Tracking", CV_WINDOW_AUTOSIZE);
	createTrackbar("maxH", "Finger Tracking", &maxHAll, 255);
	createTrackbar("minH", "Finger Tracking", &minHAll, 255);
	createTrackbar("maxS", "Finger Tracking", &maxSAll, 255);
	createTrackbar("minS", "Finger Tracking", &minSAll, 255);
	createTrackbar("maxV", "Finger Tracking", &maxVAll, 255);
	createTrackbar("minV", "Finger Tracking", &minVAll, 255);
	//createTrackbar("Size", "Finger Tracking", &size, 10000);

	while (1)
	{
		webcam.read(img);
		flip(img, img, 1);
		cvtColor(img, img_hsv, CV_BGR2HSV);
		inRange(img_hsv, Scalar(minHAll, minSAll, minVAll), Scalar(maxHAll, maxSAll, maxVAll), thresh_finger);
		//inRange(img_hsv, Scalar(minHR, minSR, minVR), Scalar(maxHR, maxSR, maxVR), thresh_red);
		//inRange(img_hsv, Scalar(minHB, minSB, minVB), Scalar(maxHB, maxSB, maxVB), thresh_blue);
		/*erode(thresh_blue, thresh_blue, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		dilate(thresh_blue, thresh_blue, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		dilate(thresh_blue, thresh_blue, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		erode(thresh_blue, thresh_blue, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		erode(thresh_red, thresh_red, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		dilate(thresh_red, thresh_red, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		dilate(thresh_red, thresh_red, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		erode(thresh_red, thresh_red, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));*/
		erode(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		dilate(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		//imshow("Blue Blob", thresh_blue);
		imshow("Finger Blob", thresh_finger);
		//imshow("Red Blob", thresh_red);
		imshow("Webcam Stream", img);
		if (waitKey(30) == 32)
		{
			break;
		}
	}
}

void count_bots()
{
	cam.read(img);
	waitKey(50);
	cam.read(img);
	waitKey(50);
	cam.read(img);
	cvtColor(img, img_hsv, CV_BGR2HSV);
	inRange(img_hsv, Scalar(minHG, minSG, minVG), Scalar(maxHG, maxSG, maxVG), blobg);
	//inRange(img_hsv, Scalar(minHR, minSR, minVR), Scalar(maxHR, maxSR, maxVR), thresh_yellow);
	erode(blobg, blobg, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	dilate(blobg, blobg, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	dilate(blobg, blobg, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	erode(blobg, blobg, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	//imshow("Blue Blob", blobg);
	//waitKey(10);

	findContours(blobg, contours_green, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	n_bots = 0;

	for (int i = 0; i < contours_green.size(); i++)
	{
		if (contours_green[i].size() > 6)
		{
			n_bots++;
		}
	}
	cout << "No of Bots Detected = " << n_bots;
	Sleep(1000);
}

int get_user_choice()
{
	int counter = 0, blob_count = 0, blob_count_old = 0;
	int draw_flag = 0, exit;
	drawing1 = Mat::zeros(img.size(), CV_8UC3);
	drawing2 = Mat::zeros(img.size(), CV_8UC3);
	drawing3 = Mat::zeros(img.size(), CV_8UC3);
	//rectangle(drawing3, Point(60, 25), Point(260, 215), Scalar(255, 255), 3);
	//drawing4 = Mat::zeros(img.size(), CV_8UC3);
	new_c = Point(-1, -1);
	old_c = Point(-1, -1);
	namedWindow("Finger Blob", CV_WINDOW_AUTOSIZE);
	namedWindow("Webcam Stream", CV_WINDOW_AUTOSIZE);
	while (1)
	{
		blob_count = 0;
		drawing1 = Mat::zeros(img.size(), CV_8UC3);
		webcam.read(img2);
		flip(img2, img2, 1);
		cvtColor(img2, img_hsv, CV_BGR2HSV);
		inRange(img_hsv, Scalar(minHAll, minSAll, minVAll), Scalar(maxHAll, maxSAll, maxVAll), thresh_finger);
		erode(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		dilate(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		imshow("Finger Blob", thresh_finger);
		Moments mom = moments(thresh_finger);
		findContours(thresh_finger, all_col, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);
		
		for (int i = 0; i < all_col.size(); i++)
		{
			//cout << "Cont: " << all_col[i] << " " << all_col[i].size();
			if (all_col[i].size() > 15)
			{
				blob_count += 1;
			}
			all_col[i].clear();
		}

		if (blob_count > 0)
		{
			if (blob_count == blob_count_old)
			{
				//cout << "Blobs: " << blob_count << ", counter = " << counter << endl;
				counter += 1;
			}
			else
			{
				counter = 0;
			}
		}
		else
		{
			//cout << "Blobs: " << blob_count << ", counter = " << counter << endl;
			counter = 0;
		}
		blob_count_old = blob_count;
		//flip(drawing1, drawing3, 1);
		//flip(drawing2, drawing4, 1);
		//flip(img2, img2, 1);
		//flip(thresh_finger, thresh_finger, 1);
		imshow("Webcam Stream", img2);

		//imshow("YY", drawing4);
		//imshow("XX", drawing1);
		exit = waitKey(10);
		if (exit >= 0x31 && exit <= 0x39)
		{
			cout << "\nYou have entered " << (exit & 0x0f) << endl;
			Sleep(1000);
			return (exit & 0x0f);
		}
		if (counter == 15)
		{
			cout << "\nYou have entered " << blob_count << endl;
			Sleep(1000);
			return blob_count;
		}
	}
	//waitKey(0);
}

void get_hand_out()
{
	int counter = 0, blob_count = 0, blob_count_old = 0;
	int draw_flag = 0, exit;
	drawing1 = Mat::zeros(img.size(), CV_8UC3);
	drawing2 = Mat::zeros(img.size(), CV_8UC3);
	drawing3 = Mat::zeros(img.size(), CV_8UC3);
	//rectangle(drawing3, Point(60, 25), Point(260, 215), Scalar(255, 255), 3);
	//drawing4 = Mat::zeros(img.size(), CV_8UC3);
	new_c = Point(-1, -1);
	old_c = Point(-1, -1);
	namedWindow("Finger Blob", CV_WINDOW_AUTOSIZE);
	namedWindow("Webcam Stream", CV_WINDOW_AUTOSIZE);
	blob_count = 0;
	drawing1 = Mat::zeros(img.size(), CV_8UC3);
	webcam.read(img2);
	flip(img2, img2, 1);
	cvtColor(img2, img_hsv, CV_BGR2HSV);
	inRange(img_hsv, Scalar(minHAll, minSAll, minVAll), Scalar(maxHAll, maxSAll, maxVAll), thresh_finger);
	erode(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	dilate(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	dilate(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	erode(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	imshow("Finger Blob", thresh_finger);
	Moments mom = moments(thresh_finger);
	findContours(thresh_finger, all_col, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	for (int i = 0; i < all_col.size(); i++)
	{
		//cout << "Cont: " << all_col[i] << " " << all_col[i].size();
		if (all_col[i].size() > 15)
		{
			blob_count += 1;
		}
		all_col[i].clear();
	}

	if (blob_count > 0)
	{
		if (blob_count == blob_count_old)
		{
			//cout << "Blobs: " << blob_count << ", counter = " << counter << endl;
			counter += 1;
		}
		else
		{
			counter = 0;
		}
	}
	else
	{
		//cout << "Blobs: " << blob_count << ", counter = " << counter << endl;
		counter = 0;
	}
	blob_count_old = blob_count;
	//flip(drawing1, drawing3, 1);
	//flip(drawing2, drawing4, 1);
	//flip(img2, img2, 1);
	//flip(thresh_finger, thresh_finger, 1);
	//imshow("Webcam Stream", img2);

	//imshow("YY", drawing4);
	//imshow("XX", drawing1);
	//exit = waitKey(10);
	/*if (exit >= 0x31 && exit <= 0x39)
	{
		cout << "\nYou have entered " << blob_count << endl;
		Sleep(2000);
		return (exit & 0x0f);
	}
	if (counter == 15)
	{
		cout << "\nYou have entered " << blob_count << endl;
		Sleep(2000);
		return blob_count;
	}*/

	if (blob_count > 0)
	{
		clear_screen();
		cout << "Please move your hand off the screen.\n";
		while (1)
		{
			blob_count = 0;
			drawing1 = Mat::zeros(img.size(), CV_8UC3);
			webcam.read(img2);
			flip(img2, img2, 1);
			cvtColor(img2, img_hsv, CV_BGR2HSV);
			inRange(img_hsv, Scalar(minHAll, minSAll, minVAll), Scalar(maxHAll, maxSAll, maxVAll), thresh_finger);
			erode(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
			dilate(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
			dilate(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
			erode(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
			//imshow("Finger Blob", thresh_finger);
			Moments mom = moments(thresh_finger);
			findContours(thresh_finger, all_col, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

			for (int i = 0; i < all_col.size(); i++)
			{
				//cout << "Cont: " << all_col[i] << " " << all_col[i].size();
				if (all_col[i].size() > 15)
				{
					blob_count += 1;
				}
				all_col[i].clear();
			}
			//flip(drawing1, drawing3, 1);
			//flip(drawing2, drawing4, 1);
			//flip(img2, img2, 1);
			//flip(thresh_finger, thresh_finger, 1);
			//imshow("Webcam Stream", img2);

			//imshow("YY", drawing4);
			//imshow("XX", drawing1);
			//waitKey(0);
			if (blob_count > 0)
			{
				clear_screen();
				cout << "Please move your hand off the screen. \n";
			}
			else
			{
				clear_screen();
				break;
			}
		}
	}
}

void locate_bots_temp()
{
	for (int i = 0; i < n_bots; i++)
	{
		locbotg[i] = Point(0, 0);
		locbotg_temp[i] = Point(0, 0);
		locr_temp[i] = Point(0, 0);
		locr[i] = Point(0, 0);
	}
	cout << "Clicking img for 1st location (Temp to Old)" << endl;
	Sleep(1000);
	
	img2 = Mat::zeros(img.size(), CV_8UC3);
	cam.read(img);
	waitKey(50);
	cam.read(img);
	waitKey(50);
	cam.read(img);
	cvtColor(img, img_hsv, CV_BGR2HSV);
	inRange(img_hsv, Scalar(minHG, minSG, minVG), Scalar(maxHG, maxSG, maxVG), blobg);
	//inRange(img_hsv, Scalar(minHR, minSR, minVR), Scalar(maxHR, maxSR, maxVR), thresh_yellow);
	erode(blobg, blobg, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	dilate(blobg, blobg, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	dilate(blobg, blobg, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	erode(blobg, blobg, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	
	findContours(blobg, contours_green, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));


	//cont_app.resize(contours_green.size());
	int xs = 0;
	int ys = 0;

	//cout << " (I think) Resized Vector Contours Size = " << contours.size() << endl;
	//system("pause");
	//cout << "Temp Locs ";
	for (int i = 0; i < contours_green.size(); i++)
	{
		if (contours_green[i].size() > 6)
		{
			//cout << "Contour " << i << "= " << contours_green[i]<<endl;
			//approxPolyDP(contours_green[i], contours_green[i], 6, true);
			//cout << "Cont App " << i << "= " << cont_app[i] << endl;
			//system("pause");
			//cout << "X = " << locbotg_temp[i].x << endl;
			//cout << "Y = " << locbotg_temp[i].y << endl;
			/*for (int j = 0; j < contours_green[i].size(); j++)
			{
				xs += contours_green[i][j].x;
				ys += contours_green[i][j].y;
			}
			locbotg_temp[i] = Point((xs / contours_green[i].size()), (ys / contours_green[i].size()));*/
			Moments mt = moments(contours_green[i]);
			locbotg_temp[i] = Point((mt.m10 / mt.m00), (mt.m01 / mt.m00));
			//cout << "Point = " << locbotg_temp[i] << endl;
			//system("pause");
			//cout << locbotg_temp[i] << ", ";
			for (int j = 0; j < n_bots; j++)
			{
				if (locbotg_old[j] == Point(0,0))
				{
					locbotg_old[j] = locbotg_temp[i];
					//locbotg_temp[i] = Point(0, 0);
					break;
				}
			}
			
			contours_green[i].clear();
			//cont_app[i].clear();
			//circle(img2, locbotg_temp[i], 1, Scalar(0, 0, 255), -1);
		}
	}

	//cout << endl;
	//system("pause");
	img2 += img;
	namedWindow("Stream", CV_WINDOW_AUTOSIZE);
	imshow("Stream", img2);
	waitKey(10);

}

void calib_id_quick()
{

	//namedWindow("Green Blobs", CV_WINDOW_AUTOSIZE);
	//cout << "\nOld Locations: ";
	for (int i = 0; i < n_bots; i++)
	{
		cout << locbotg_old[i]<<" ";
	}
	//cout << endl;

	for (int id = 0; id < n_bots; id++)
	{
		clear_screen();
		cout << "Displacing Bot with ID No " << id + 1 << "... " << endl;
		Sleep(500);
		
		com.send_data(id + 1);
		Sleep(5);
		com.send_data(0x06);
		Sleep(5);
		com.send_data(250);
		Sleep(5);
		com.send_data(250);
		Sleep(5);

		waitKey(1000);

		com.send_data(0);
		Sleep(5);
		com.send_data(0);
		Sleep(5);
		com.send_data(0);
		Sleep(5);
		com.send_data(0);
		Sleep(5);

		img2 = Mat::zeros(img.size(), CV_8UC3);
		for (int i = 0; i < 5; i++)
		{
			cam.read(img);
			waitKey(30);
		}

		cvtColor(img, img_hsv, CV_BGR2HSV);
		inRange(img_hsv, Scalar(minHG, minSG, minVG), Scalar(maxHG, maxSG, maxVG), blobg);

		erode(blobg, blobg, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		dilate(blobg, blobg, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		dilate(blobg, blobg, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		erode(blobg, blobg, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

		imshow("Blue Blob", blobg);
		waitKey(10);

		findContours(blobg, contours_green, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));



		int xs = 0;
		int ys = 0;


		for (int i = 0; i < contours_green.size(); i++)
		{
			xs = 0;
			ys = 0;

			if (contours_green[i].size() > 6)
			{
				Moments mt = moments(contours_green[i]);
				locbotg_temp[i] = Point((mt.m10 / mt.m00), (mt.m01 / mt.m00));
			}
		}
		for (int i = 0; i < n_bots; i++)
		{
			/*cout << "New Positions of Robots: ";
			for (int j = 0; j < n_bots; j++)
			{
				cout << locbotg_temp[j] << ", ";
			}
			cout << endl;*/



			int count = 0;
			for (int j = 0; j < n_bots; j++)
			{
				if (((locbotg_temp[i].x - locbotg_old[j].x)*(locbotg_temp[i].x - locbotg_old[j].x)) + ((locbotg_temp[i].y - locbotg_old[j].y) * (locbotg_temp[i].y - locbotg_old[j].y)) <= 64)
				{
					count++;
				}
			}
			if (count == 0)
			{
				//cout << "\nThe Robot at Current Location: " << locbotg_temp[i] << " is Bot with ID No: " << id + 1 << endl;
				locbotg[id] = locbotg_temp[i];
				for (int j = 0; j < n_bots; j++)
				{
					locbotg_old[j] = locbotg_temp[j];
				}
				cout << endl;
				break;
			}

			contours_green[i].clear();
		}
		imshow("Stream", img);
		waitKey(10);
		Sleep(1000);

		/*cout << "\nCorrectly Identified Locations with ID No are: \n";
		for (int i = 0; i < n_bots; i++)
		{
			cout << locbotg[i];
		}
		cout << endl;*/
	}


	cam.read(img);
	waitKey(30);
	cam.read(img);
	waitKey(30);
	cam.read(img);
	waitKey(30);


	cvtColor(img, img_hsv, CV_BGR2HSV);
	inRange(img_hsv, Scalar(minHR, minSR, minVR), Scalar(maxHR, maxSR, maxVR), blobr);

	erode(blobr, blobr, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	dilate(blobr, blobr, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	dilate(blobr, blobr, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	erode(blobr, blobr, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

	findContours(blobr, contours_red, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	for (int i = 0; i < contours_red.size(); i++)
	{
		if (contours_red[i].size() > 6)
		{
			Moments mt = moments(contours_red[i]);
			locr_temp[i] = Point((mt.m10 / mt.m00), (mt.m01 / mt.m00));
			for (int j = 0; j < n_bots; j++)
			{
				if (((locr_temp[i].x - locbotg[j].x)*(locr_temp[i].x - locbotg[j].x)) + ((locr_temp[i].y - locbotg[j].y) * (locr_temp[i].y - locbotg[j].y)) <= 625)
				{
					locr[j] = locr_temp[i];
					//cout << "Location of Red for bot no " << j + 1 << ": " << locr[j]<<endl;
					//system("pause");
					break;
				}
			}
		}
		contours_red[i].clear();
	}
	destroyWindow("Stream");
	destroyWindow("Blue Blob");
}

void locate_bots_quick()
{
	//img2 = Mat::zeros(img.size(), CV_8UC3);
	
	
	cam.read(img);
	cvtColor(img, img_hsv, CV_BGR2HSV);
	inRange(img_hsv, Scalar(minHG, minSG, minVG), Scalar(maxHG, maxSG, maxVG), blobg);

	erode(blobg, blobg, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	dilate(blobg, blobg, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	dilate(blobg, blobg, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	erode(blobg, blobg, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

	findContours(blobg, contours_green, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	//int xs = 0;
	//int ys = 0;


	for (int i = 0; i < contours_green.size(); i++)
	{
		if (contours_green[i].size() > 6)
		{
			//approxPolyDP(contours_green[i], contours_green[i], 6, true);

			//xs = 0;
			//ys = 0;

			/*for (int j = 0; j < contours_green[i].size(); j++)
			{
				xs += contours_green[i][j].x;
				ys += contours_green[i][j].y;
			}*/
			Moments mt = moments(contours_green[i]);
			locbotg_temp[i] = Point((mt.m10 / mt.m00), (mt.m01 / mt.m00));
			for (int j = 0; j < n_bots; j++)
			{
				if (((locbotg_temp[i].x - locbotg[j].x)*(locbotg_temp[i].x - locbotg[j].x)) + ((locbotg_temp[i].y - locbotg[j].y) * (locbotg_temp[i].y - locbotg[j].y)) <= 400)
				{
					locbotg[j] = locbotg_temp[i];
					//e_x_bots[j] = goals[j].x - locbotg[j].x;
					//e_y_bots[j] = goals[j].y - locbotg[j].y;
					//e_x_sum[i] += e_x_bots[i];
					//e_y_sum[i] += e_y_bots[i];
					circle(img, locbotg[j], 23, Scalar(0, 255, 0), 1);
					circle(img, locbotg[j], 30, Scalar(0, 0, 255), 1);
					break;
				}
			}
		}
		imshow("Stream", img);
		contours_green[i].clear();
	}
}

void get_xy_errors(int id)
{
	e_x_bots = goals[id].x - locbotg[id].x;
	e_y_bots = goals[id].y - locbotg[id].y;
	e_x_sum[id] += e_x_bots;
	e_y_sum[id] += e_y_bots;
}

void get_angles()
{
	inRange(img_hsv, Scalar(minHR, minSR, minVR), Scalar(maxHR, maxSR, maxVR), blobr);

	erode(blobr, blobr, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	dilate(blobr, blobr, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	dilate(blobr, blobr, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	erode(blobr, blobr, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

	findContours(blobr, contours_red, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
	
	for (int i = 0; i < contours_red.size(); i++)
	{
		if (contours_red[i].size() > 6)
		{
			Moments mt = moments(contours_red[i]);
			locr_temp[i] = Point((mt.m10 / mt.m00), (mt.m01 / mt.m00));
			for (int j = 0; j < n_bots; j++)
			{
				if (((locr_temp[i].x - locr[j].x)*(locr_temp[i].x - locr[j].x)) + ((locr_temp[i].y - locr[j].y) * (locr_temp[i].y - locr[j].y)) <= 225)
				{
					locr[j] = locr_temp[i];
					if (locbotg[j].x == locr[j].x)
					{
						if (locr[j].y > locbotg[j].y)
						{
							phi_bots[j] = 90;
						}
						else
						{
							phi_bots[j] = -90;
						}
					}
					else
					{
						phi_bots[j] = (atan2((locr[j].y - locbotg[j].y), (locr[j].x - locbotg[j].x)) * 180) / 3.14;
					}
					break;
				}
			}
		}
		contours_red[i].clear();
		//line(img, locbotg[i], locr[i], Scalar(255, 0, 0));
	}
}

void get_goal_angle(int id)
{
	if (locbotg[id].x == goals[id].x)
	{
		if (locbotg[id].y > goals[id].y)
		{
			phi_g = -90;
		}
		else
		{
			phi_g = 90;
		}
	}
	else
	{
		phi_g = (atan2((goals[id].y - locbotg[id].y), (goals[id].x - locbotg[id].x)) * 180) / 3.14;
	}
	e_phi = angle_in_range(phi_g - phi_bots[id]);
	e_phi_sum[id] += e_phi;

	circle(img, goals[id], 2, Scalar(0, 255, 255));
	imshow("Stream", img);
}

/*void correct_location_all()
{
	for (int i = 0; i < n_bots; i++)
	{
		get_goal_angles(i);
		int omega = (24 * e_phi[i]) + (1 * e_phi_sum[i]);
		float x = (20 * e_x_bots[i]) + (1 * e_x_sum[i]);// +(0.01*(e_x - ex_old));
		float y = (20 * e_y_bots[i]) + (1 * e_y_sum[i]);// + (0.01*(e_y - ey_old));
		vel = sqrt((x*x) + (y*y));
		unsigned char vl = (unsigned char)((2 * vel - 15.5*omega) / 5);
		unsigned char vr = (unsigned char)((2 * vel + 15.5*omega) / 5);
		if (vr > 220)
		{
			vr = 220;
		}
		else if (vr < 170)
		{
			vr = 170;
		}
		if (vl > 220)
		{
			vl = 220;
		}
		else if (vl < 170)
		{
			vl = 170;
		}
		unsigned char motion;
		if (e_phi[i] > 10)
		{
			motion = 0x02;
		}
		else if (e_phi[i] < -10)
		{
			motion = 0x04;
		}
		else
		{
			motion = 0x06;
		}
		if (((e_x_bots[i] < 15 && e_x_bots[i] > -15) && (e_y_bots[i] < 15 && e_y_bots[i] > -15)))
		{
			motion = 0x00;
			breaker = 0;
		}
		com.send_data(i + 1);
		waitKey(1);
		com.send_data(motion);
		waitKey(1);
		com.send_data(vl);
		waitKey(1);
		com.send_data(vr);
		waitKey(1);


		//cout <<"Phi Bot: "<< phi_bot << "Error: " << e << ", w: " << omega << ", Motion: " << (int)motion << ", Vl: " << (int)vl << ", Vr: " << (int)vr << endl;
		//cout << "x_bl: " << nx_b << ", y_bl: " << ny_b << ", x_yell: " << nx_p << ", y_yell: " << ny_p << ", Phi Bot: " << phi_bot << ", Phi Goal: " << phi_g << endl;
		//circle(img, Point(x_g[i], y_g[i]), 3, Scalar(0, 255, 255));
		//circle(img, Point(bot_x, bot_y), 25, Scalar(0, 255, 0));
		//circle(img, Point(x_o, y_o), 10, Scalar(0, 0, 255));
		//imshow("Blue Blob", thresh_blue);
		//imshow("Yellow Blob", thresh_yellow);
		//imshow("Stream", img);
	}
}*/

void go_to_goal_ind(int id)
{
	get_xy_errors(id);
	int omega = (24 * e_phi) + (1 * e_phi_sum[id]);
	float x = (20 * e_x_bots) + (1 * e_x_sum[id]);// +(0.01*(e_x - ex_old));
	float y = (20 * e_y_bots) + (1 * e_y_sum[id]);// + (0.01*(e_y - ey_old));
	vel = sqrt((x*x) + (y*y));
	unsigned char vl = (unsigned char)((2 * vel - 15.5*omega) / 5);
	unsigned char vr = (unsigned char)((2 * vel + 15.5*omega) / 5);
	if (vr > 200)
	{
		vr = 200;
	}
	else if (vr < 170)
	{
		vr = 170;
	}
	if (vl > 200)
	{
		vl = 200;
	}
	else if (vl < 170)
	{
		vl = 170;
	}
	unsigned char motion;
	if (e_phi > 30)
	{
		motion = 0x0a;
	}
	else if (e_phi < -30)
	{
		motion = 0x05;
	}
	else if (e_phi > -30 && e_phi < -10)
	{
		motion = 0x04;
	}
	else if (e_phi > 10 && e_phi < 30)
	{
		motion = 0x02;
	}
	else
	{
		motion = 0x06;
	}
	if (((e_x_bots < 5 && e_x_bots > -5) && (e_y_bots < 5 && e_y_bots > -5)))
	{
		motion = 0x00;
		reached |= (0x01 << id);
	}
	else
	{
		reached &= ~(0x01 << id); 
	}
	if (motion != 0)
	{
		com.send_data(id + 1);
		Sleep(5);
		com.send_data(motion);
		Sleep(5);
		com.send_data(vl);
		Sleep(5);
		com.send_data(vr);
		Sleep(5);
	}
	else
	{
		com.send_data(id + 1);
		Sleep(5);
		com.send_data(motion);
		Sleep(5);
	}
	
}

void get_obst_angle(int id)
{
	if (locbotg[id].x == obstacles[id].x)
	{
		if (locbotg[id].y > obstacles[id].y)
		{
			phi_obst = -90;
		}
		else
		{
			phi_obst = 90;
		}
	}
	else
	{
		phi_obst = (atan2((obstacles[id].y - locbotg[id].y), (obstacles[id].x - locbotg[id].x)) * 180) / 3.14;
	}
	e_obst = angle_in_range(phi_g  - phi_obst);
	//e_phi_sum[id] += e_phi[id];
}

int bot_goal_distance(int id)
{
	return (((locbotg[id].x - goals[id].x) * (locbotg[id].x - goals[id].x)) + ((locbotg[id].y - goals[id].y) * (locbotg[id].y - goals[id].y)));
}

int bot_obst_distance(int id)
{
	return (((locbotg[id].x - obstacles[id].x) * (locbotg[id].x - obstacles[id].x)) + ((locbotg[id].y - obstacles[id].y) * (locbotg[id].y - obstacles[id].y)));
}

void avoid_obstacle_ind(int id)
{
	if (obst_flag[id] == 1)
	{
		get_obst_angle(id);
		if (fabs((double)e_obst) < 90)
		{
			if ((bot_goal_distance(id) < bot_obst_distance(id)) && (fabs((double)e_obst) < 20))
			{
				motion = 0x00;
				com.send_data(id + 1);
				Sleep(5);
				com.send_data(motion);
				Sleep(5);
			}
			else
			{
				if (e_obst < 0)
				{
					/*if (e_obst <= 15 && e_obst >= 0)
					{
						e_obst = 45;
					}
					else if (e_obst >= -15 && e_obst < 0)
					{
						e_obst = -45; 
					}*/
					if ((phi_bots[id] <= (fabs((double)e_obst) + phi_g)) && (phi_bots[id] >(fabs((double)e_obst) - phi_g)))
					{
						l = 1;
						r = 0;
						motion = 0x04;
					}
					else
					{
						r = 1;
						l = 0;
						motion = 0x0a;
					}
				}
				else
				{
					if ((phi_bots[id] <= (fabs((double)e_obst) + phi_g)) && (phi_bots[id] > (fabs((double)e_obst) - phi_g)))
					{
						r = 1;
						l = 0;
						motion = 0x02;
					}
					else
					{
						l = 1;
						r = 0;
						motion = 0x05;
					}
				}
				/*if (r == 1)
				{
					motion = 0x02;
				}
				else
				{
					motion = 0x04;
				}*/
				com.send_data(id + 1);
				Sleep(5);
				com.send_data(motion);
				Sleep(5);
				com.send_data(180);
				Sleep(5);
				com.send_data(180);
				Sleep(5);
			}
		}
		else
		{
			go_to_goal_ind(id);
		}
	}
	else
	{
		motion = 0x00;
		com.send_data(id+1);
		Sleep(5);
		com.send_data(motion);
		Sleep(5);
	}
}

void check_env()
{
	for (int i = 0; i < n_bots; i++)
	{
		count_ob = 0;
		for (int j = 0; j < n_bots; j++)
		{
			if (((((locbotg[i].x - locbotg[j].x)*(locbotg[i].x - locbotg[j].x)) + ((locbotg[i].y - locbotg[j].y) * (locbotg[i].y - locbotg[j].y))) <= 3800) && (i != j))
			{
				count_ob++;
				if (count_ob == 1)
				{
					obstacles[i] = locbotg[j];
					//cout << "Ost of " << i + 1 << "= " << obstacles[i] << endl;
				}
			}
		}
		obst_flag[i] = count_ob;
	}
}

void form_shape()
{
	namedWindow("Stream", CV_WINDOW_AUTOSIZE);
	reached = 0;
	int exit;
	int reached_all = 0;
	for (int i = 0; i < n_bots; i++)
	{
		reached_all |= (0x01 << i);
	}
	//cout << "reached_all = " << reached_all;
	//system("pause");
	while (1)
	{
		locate_bots_quick();
		get_angles();
		check_env();
		for (int i = 0; i < n_bots; i++)
		{
			/*if (i == 0)
			{
				com.send_data(n_bots);
				Sleep(2);
				com.send_data(0x00);
				Sleep(2);
				com.send_data(n_bots);
				Sleep(2);
				com.send_data(0x00);
				Sleep(2);
				com.send_data(n_bots);
				Sleep(2);
				com.send_data(0x00);
				Sleep(2);
				com.send_data(0x00);
				//waitKey(0);
			}
			else
			{
				com.send_data(i);
				Sleep(2);
				com.send_data(0x00);
				Sleep(2);
				com.send_data(i);
				Sleep(2);
				com.send_data(0x00);
				Sleep(2);
				com.send_data(i);
				Sleep(2);
				com.send_data(0x00);
				Sleep(2);
				com.send_data(0x00);
				//waitKey(1);
			}*/
			//waitKey(0);
			get_goal_angle(i);
			if ((obst_flag[i] < 1) )
			{
				go_to_goal_ind(i);
			}
			else
			{
				avoid_obstacle_ind(i);
			}
			//cout << endl;
			//waitKey(300);
		}
		exit = waitKey(10);
		if ((reached == reached_all) || exit == 32 || exit == 27)
		{
			if (exit == 27)
			{
				exit_flag = 1;
			}
			break;
		}
		/*if (breaker == 0)
		{
		break;
		}*/
		//get_goal_angles(0);
		//show_contours_red_quick();
	}
	//for (int i = 1; i <= n_bots; i++)
	//{
		com.send_data(0);
		Sleep(5);
		com.send_data(0);
		Sleep(5);
		com.send_data(0);
		Sleep(5);
		com.send_data(0);
		Sleep(5);

		//blank = database[pred - 1];
		//imshow("Stream", img + blank);
		//waitKey(0);
		//com.send_data(0x00);
		//waitKey(1);
		//com.send_data(0x00);
		//waitKey(1);
	//}
	/*for (i = 0; i < 3; i++)
	{
	while (end_trav == 1)
	{
	if (bot_obst_dist() > 900)
	{
	go_to_goal();
	}
	else
	{

	avoid_obstacle();
	init_errors();
	}
	}
	}*/
	/*com.send_data(0x00);
	waitKey(1);
	com.send_data(0x00);
	waitKey(1);
	com.send_data(0x00);
	waitKey(1);
	com.send_data(0x00);
	waitKey(1);
	com.send_data(0x00);
	waitKey(1);
	com.send_data(0x00);
	waitKey(1);
	end_trav = 1;
	breaker = 1;*/
	//}
	/*while (1)
	{
		cam.read(img);
		imshow("Stream", img);
		if (waitKey(30) == 27)
		{
			break;
		}
	}*/
}

void recognise_gesture()
{
	int exit;
	int counter1 = 0, c_flag = 0, counter2 = 0;
	Point new_c = Point(-1, -1);
	Point old_c = Point(-1, -1);
	Point cnew_c = Point(-1, -1);
	Point cold_c = Point(-1, -1);
	Mat drawing1 = Mat::zeros(img.size(), CV_8UC3);
	Mat drawing2 = Mat::zeros(img.size(), CV_8UC3);
	Mat drawing3 = Mat::zeros(Size(20, 20), CV_8UC1);
	Mat drawing4 = Mat::zeros(Size(20, 20), CV_8UC1);
	cam.read(img2);
	namedWindow("Webcam Stream", CV_WINDOW_AUTOSIZE);
	namedWindow("Stream", CV_WINDOW_AUTOSIZE);
	namedWindow("Finger Blob", CV_WINDOW_AUTOSIZE);
	imshow("Stream", img2);
	webcam.read(img);
	webcam.read(img);
	webcam.read(img);
	waitKey(2000);

	while (1)
	{
		webcam.read(img);
		flip(img, img, 1);
		cvtColor(img, img_hsv, CV_BGR2HSV);
		inRange(img_hsv, Scalar(minHFing, minSFing, minVFing), Scalar(maxHFing, maxSFing, maxVFing), thresh_finger);
		erode(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		dilate(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		Moments mom = moments(thresh_finger);
		//cout << mom.m00 << endl;
		if (mom.m00 > size)
		{
			//c_flag = 1;
			counter2 = 0;
			new_c = Point((mom.m10 / mom.m00), (mom.m01 / mom.m00));
			cnew_c = Point((mom.m10 / (16 * mom.m00)), (mom.m01 / (12 * mom.m00)));
			if (old_c.x > -1)
			{
				circle(drawing1, new_c, 5, Scalar(0, 255, 0), 1);
				line(drawing2, new_c, old_c, Scalar(255, 255, 0), 3);
				line(drawing3, cnew_c, cold_c, Scalar(255), 1);
				if (point_distance(new_c, old_c) < 25)
				{
					counter1 += 1;
					//cout << counter1 << endl;
				}
				else
				{
					counter1 = 0;
				}
			}
			old_c = new_c;
			cold_c = cnew_c;
			drawing1 = Mat::zeros(img.size(), CV_8UC3);
		}
		else
		{
			if (old_c.x > -1)
			{
				if (new_c == old_c)
				{
					counter2 += 1;
					//cout << counter2 << endl;
				}
			}
			/*if (c_flag == 1)
			{
				counter2 += 1;
				cout << counter2 << endl;
			}*/
			
		}
		
		imshow("Webcam Stream", img);
		imshow("Stream", img2 + drawing2 + drawing1);
		imshow("Finger Blob", thresh_finger);
		//imshow("YY", drawing4);
		//imshow("XX", drawing1);
		exit = waitKey(10);
		if (exit == 32 || exit == 27 || counter1 == 15 || counter2 == 15)
		{
			if (exit == 27)
			{
				exit_flag = 1;
			}
			break;
		}
		if (exit == 'c')
		{
			old_c = Point(-1, -1);
			new_c = Point(-1, -1);
			cold_c = Point(-1, -1);
			cnew_c = Point(-1, -1);
			drawing2 = Mat::zeros(img.size(), CV_8UC3);
			drawing3 = Mat::zeros(Size(20, 20), CV_8UC1);
		}
	}
	destroyWindow("Webcam Stream");
	destroyWindow("Finger Blob");
	imwrite("C:/Users/ABHISHEK/Desktop/Abhishek Stuff/Robotics/Online Courses/Machine Learning/Programming Exercises/Week 5/mlclass-ex4-006/mlclass-ex4/BE Project/MatOCV Comm/Gesture Input.jpg", drawing3);
	ofstream flagfile;
	ifstream opfile;
	flagfile.open("C:/Users/ABHISHEK/Desktop/Abhishek Stuff/Robotics/Online Courses/Machine Learning/Programming Exercises/Week 5/mlclass-ex4-006/mlclass-ex4/BE Project/MatOCV Comm/flag.txt");
	flagfile << "1";
	flagfile.close();
	waitKey(500);
	opfile.open("C:/Users/ABHISHEK/Desktop/Abhishek Stuff/Robotics/Online Courses/Machine Learning/Programming Exercises/Week 5/mlclass-ex4-006/mlclass-ex4/BE Project/MatOCV Comm/shape_prediction.txt");
	opfile >> pred;
	opfile.close();
	clear_screen();
	if (exit_flag == 0)
	{
		cout << "Your gesture has been identified as: ";
		switch (pred)
		{
		case 1: cout << "V\n";
			Sleep(2000);
			break;
		case 2: cout << "Rectangle\n";
			Sleep(2000);
			break;
		case 3: cout << "Circle\n";
			Sleep(2000);
			break;
		case 4: cout << "T\n";
			Sleep(2000);
			break;
		case 5: cout << "Inverted V\n";
			Sleep(2000);
			break;
		case 6: cout << "Line\n";
			Sleep(2000);
			break;
		}
	}
}

int bot_goal_dist(int g_no, int id)
{
	return (((locbotg[id].x - shapes[pred - 1][g_no].x) * (locbotg[id].x - shapes[pred - 1][g_no].x)) + ((locbotg[id].y - shapes[pred - 1][g_no].y) * (locbotg[id].y - shapes[pred - 1][g_no].y)));
}

void assign_goals()
{
	cam.read(img);
	cam.read(img);
	
	//img = Mat::zeros(Size(320, 240), CV_8UC3);
	img2 = img.clone();
	for (int i = 0; i < n_bots; i++)
	{
		circle(img, shapes[pred - 1][i], 3, Scalar(0, 255, 255), 1);
	}
	imshow("Stream", img);
	waitKey(1000);
	int bot_goal_distances[100][100] = { 0 };
	int min_dist;
	int min_id;
	for (int i = 0; i < n_bots; i++)		//bot
	{
		for (int j = 0; j < n_bots; j++)	//goal
		{
			bot_goal_distances[i][j] = bot_goal_dist(j, i);
		}
		taken[i] = 0;
	}
	
	int tot_dist[100] = { 0 };
	int id_sort[100] = { 0 };
	for (int i = 0; i < n_bots; i++)
	{
		for (int j = 0; j < n_bots; j++)
		{
			tot_dist[i] += bot_goal_distances[j][i];
		}
		id_sort[i] = i;
	}

	//sorting

	for (int i = 0; i < n_bots; i++)
	{
		for (int j = 0; j < n_bots - 1; j++)
		{
			if (tot_dist[j] < tot_dist[j + 1])
			{
				int t = tot_dist[j];
				tot_dist[j] = tot_dist[j + 1];
				tot_dist[j + 1] = t;
				t = id_sort[j];
				id_sort[j] = id_sort[j + 1];
				id_sort[j + 1] = t;
			}
		}
	}

	/*for (int i = 0; i < n_bots; i++)
	{
		cout << "Dist for goal: " << i + 1 << " = " << tot_dist[i] << endl;
	}
	waitKey(0);*/

	Scalar color;
	for (int k = 0; k < 1; k++)
	{
		for (int i = 0; i < n_bots; i++)
		{
			img = img2.clone();
			int min_dist = point_distance(Point(0,0), Point(320, 320));
			int min_id = 0;
			if (i == 0)
			{
				color = Scalar(255, 0, 255);
			}
			else if (i == 1)
			{
				color = Scalar(0, 255, 0);
			}
			else if (i == 2)
			{
				color = Scalar(0, 0, 255);
			}
			if (i == 3)
			{
				color = Scalar(0, 255, 255);
			}
			if (i == 4)
			{
				color = Scalar(255, 255, 0);
			}
			for (int j = 0; j < n_bots; j++)
			{
				line(img, shapes[pred - 1][id_sort[i]], locbotg[j], color, 1);
				if ((bot_goal_dist(id_sort[i], j) < min_dist) && (taken[j] == 0))
				{
					min_dist = bot_goal_dist(id_sort[i], j);
					min_id = j;
				}
			}
			imshow("Stream", img);
			waitKey(300);
			goals[min_id] = shapes[pred - 1][id_sort[i]];
			taken[min_id] = 1;
		}
	}
	img = img2.clone();
	for (int i = 0; i < n_bots; i++)
	{
		int min_dist = tot_dist[0];
		int min_id = 0;
		if (i == 0)
		{
			color = Scalar(255, 0, 255);
		}
		else if (i == 1)
		{
			color = Scalar(0, 255, 0);
		}
		else if (i == 2)
		{
			color = Scalar(0, 0, 255);
		}
		if (i == 3)
		{
			color = Scalar(0, 255, 255);
		}
		if (i == 4)
		{
			color = Scalar(255, 255, 0);
		}
		line(img, goals[i], locbotg[i], color, 1);
	}
	imshow("Stream", img);
	waitKey(2000);
}

void assign_goals_2()
{
	Scalar color;
	double min_sum = (5 * point_distance(Point(320, 320), Point(0, 0)));
	cout << "Min Sum = " << min_sum << endl;
	Sleep(1000);
	for (int i = 0; i < n_bots; i++)
	{
		circle(blank, locbotg[i], 3, Scalar(0, 0, 255));
	}
	Mat blank2 = blank.clone();
	namedWindow("Stream2", CV_WINDOW_AUTOSIZE);
	for (int a = 0; a < n_bots; a++)
	{
		for (int b = 0; b < n_bots; b++)
		{		
			for (int c = 0; c < n_bots; c++)
			{
				for (int d = 0; d < n_bots; d++)
				{
					for (int e = 0; e < n_bots; e++)
					{
						//cout << goal_temp[a] << ", " << goal_temp[b] << ", " << goal_temp[c] << ", " << goal_temp[d] << ", " << goal_temp[e] << endl;
						//cout << a << ", " << b << ", " << c << ", " << d << ", " << e << endl;
						if ((a != b) && (a != c) && (a != d) && (a != e) && (b != c) && (b != d) && (b != e) && (c != d) && (c != e) && (d != e))
						{
							goal_temp[0] = shapes[pred - 1][a];
							goal_temp[1] = shapes[pred - 1][b];
							goal_temp[2] = shapes[pred - 1][c];
							goal_temp[3] = shapes[pred - 1][d];
							goal_temp[4] = shapes[pred - 1][e];
							int sum_temp = 0;
							for (int i = 0; i < n_bots; i++)
							{
								sum_temp += point_distance(goal_temp[i], locbotg[i]);
								if (i == 0)
								{
									color = Scalar(255, 0, 255);
								}
								else if (i == 1)
								{
									color = Scalar(0, 255, 0);
								}
								else if (i == 2)
								{
									color = Scalar(0, 0, 255);
								}
								if (i == 3)
								{
									color = Scalar(0, 255, 255);
								}
								if (i == 4)
								{
									color = Scalar(255, 255, 0);
								}
								line(blank, goal_temp[i], locbotg[i], color, 1);
							}
							imshow("Stream2", blank);
							waitKey(10);
							//cout << "Sum Temp = " << sum_temp << endl;
							if (sum_temp < min_sum)
							{
								goals[a] = goal_temp[a];
								goals[b] = goal_temp[b];
								goals[c] = goal_temp[c];
								goals[d] = goal_temp[d];
								goals[e] = goal_temp[e];
								min_sum = sum_temp;
								//cout << "sum_temp < min_sum...\n New Goals Assigned: \n";
								/*for (int i = 0; i < n_bots; i++)
								{
									cout << goals[i] << ", ";
								}*/
								//cout << endl;
								//imshow("Stream", blank);
								//waitKey(0);
							}
							blank = blank2.clone();
						}
						Sleep(10);
					}
				}
			}
		}
	}
	for (int i = 0; i < n_bots; i++)
	{
		line(blank, goals[i], locbotg[i], Scalar(0, 255, 0), 1);
	}
	imshow("Stream2", blank);
	waitKey(1000);
	destroyWindow("Stream2");
}

void correct_angle_ind(int id)
{
	get_goal_angle(id);
	if (e_phi > 13)
	{
		motion = 0x0a;
		com.send_data(id+1);
		Sleep(5);
		com.send_data(motion);
		Sleep(5);
		com.send_data(160);
		Sleep(5);
		com.send_data(160);
		Sleep(5);
		reached &= ~(0x01 << id);
	}
	else if (e_phi < -13)
	{
		motion = 0x05;
		com.send_data(id + 1);
		Sleep(5);
		com.send_data(motion);
		Sleep(5);
		com.send_data(160);
		Sleep(5);
		com.send_data(160);
		Sleep(5);
		reached &= ~(0x01 << id);
	}
	else
	{
		motion = 0x00;
		com.send_data(id + 1);
		Sleep(5);
		com.send_data(motion);
		Sleep(5);
		reached |= 0x01 << id;
	}
}

void correct_zoom_angles()
{
	cam.read(img);
	int exit = 0;
	reached = 0;
	int reached_all = 0;
	for (int i = 0; i < n_bots; i++)
	{
		reached_all |= 0x01 << i;
	}
	for (int i = 0; i < n_bots; i++)
	{
		goals[i] = Point(160, 120);
		if (locbotg[i].x == goals[i].x)
		{
			phi_zoom[i] = -90;
		}
		else
		{
			phi_zoom[i] = angle_in_range((atan2((locbotg[i].y - goals[i].y), (locbotg[i].x - goals[i].x))*180)/3.14);
		}
		circle(img, locbotg[i], 20, Scalar(0, 255, 0));
		imshow("Stream", img);
		waitKey(10);
		cout << "\nBOT " << i + 1 << "at " << locbotg[i] << endl;
		cout << "Goal " << i + 1 << "at " << goals[i] << endl;
		cout << "Zoom Angle for BOT " << i + 1 << ": " << phi_zoom[i] << endl;
		system("pause");
	}
	while (1)
	{
		locate_bots_quick();
		get_angles();
		for (int i = 0; i < n_bots; i++)
		{
			line(img, Point(160, 120), locbotg[i], Scalar(255, 0, 0));
			correct_angle_ind(i);
			cout << endl;
		}
		exit = waitKey(10);
		if (reached == reached_all || exit == 27)
		{
			break;
		}
	}
	com.send_data(0);
	Sleep(3);
	com.send_data(0);
	Sleep(3);
	com.send_data(0);
	Sleep(3);
	com.send_data(0);
	Sleep(3);
}

void locate_zoom_markers()
{
	namedWindow("Finger Stream", CV_WINDOW_AUTOSIZE);
	webcam.read(img);
	/*webcam.read(img);
	webcam.read(img);*/
	cvtColor(img, img_hsv, CV_BGR2HSV);
	inRange(img_hsv, Scalar(minHFing, minSFing, minVFing), Scalar(maxHFing, maxSFing, maxVFing), blobg);

	erode(blobg, blobg, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	dilate(blobg, blobg, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
	dilate(blobg, blobg, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
	erode(blobg, blobg, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));

	imshow("XX", blobg);

	findContours(blobg, contours_red, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	//int xs = 0;
	//int ys = 0;

	fing_count = 0;
	for (int i = 0; i < contours_red.size(); i++)
	{
		if (contours_red[i].size() > 10)
		{
			Moments mt = moments(contours_red[i]);
			locfing[fing_count] = Point((mt.m10 / mt.m00), (mt.m01 / mt.m00));
			fing_count += 1;
		}
		contours_red[i].clear();
	}
	imshow("Finger Stream", img);
}

int check_location_limit()
{
	int return_val = 1;
	for (int i = 0; i < n_bots; i++)
	{
		for (int j = 0; j < n_bots; j++)
		{
			if ((out_flag == 1))
			{
				if (((fabs((double)(locbotg[i].y - locbotg[j].y)) < 10) && fabs((double)(locbotg[i].x - locbotg[j].x > 120))) && ((fabs((double)(locbotg[i].x - locbotg[j].x)) < 10) && fabs((double)(locbotg[i].y - locbotg[j].y > 120))))
				{
					return_val = 1;
				}
				else
				{
					return_val = 0;
					break;
				}
			}
			else if (in_flag == 1)
			{
				if (((fabs((double)(locbotg[i].y - locbotg[j].y)) < 10) && fabs((double)(locbotg[i].x - locbotg[j].x < 280))) || ((fabs((double)(locbotg[i].x - locbotg[j].x)) < 10) && fabs((double)(locbotg[i].y - locbotg[j].y > 200))))
				{
					return_val = 1;
				}
				else
				{
					return_val = 0;
					break;
				}
			}
		}
	}
	return return_val;
}

void zoom()
{
	in_flag = 0;
	out_flag = 0;
	cam.read(img2);
	if (fing_count == 2)
	{
		fing_dist = (((locfing[0].x - locfing[1].x)*(locfing[0].x - locfing[1].x)) + ((locfing[0].y - locfing[1].y)*(locfing[0].y - locfing[1].y)));
		if (fing_dist > 30625)
		{
			out_flag = 1;
			if(check_location_limit())
			{
				for (int i = 0; i < n_bots; i++)
				{
					if (((fabs((double)(locbotg[i].x - 160)) > 6) && ((locbotg[i].x > 40) && (locbotg[i].x < 280))) || ((fabs((double)(locbotg[i].y - 120)) > 6) && ((locbotg[i].y > 40) && (locbotg[i].y < 200))))
					{
						com.send_data(i + 1);
						Sleep(3);
						com.send_data(0x06);
						Sleep(3);
						com.send_data(180);
						Sleep(3);
						com.send_data(180);
						Sleep(3);
					}
					else
					{
						com.send_data(id + 1); 
						Sleep(3);
						com.send_data(0);
						Sleep(3);
					}
						
				}
				cout << "Zooming IN\n";
			}
			else
			{
				cout << "Cannot Zoom More!!!\n";
				com.send_data(0);
				Sleep(3);
				com.send_data(0);
				Sleep(3);
				com.send_data(0);
				Sleep(3);
				com.send_data(0);
				Sleep(3);
			}
		}
		else if (fing_dist < 10000)
		{
			in_flag = 1;
			if (check_location_limit())
			{
				for (int i = 0; i < n_bots; i++)
				{
					if (((fabs((double)(locbotg[i].x - 160)) > 6) && ((locbotg[i].x > 40) && (locbotg[i].x < 280))) || ((fabs((double)(locbotg[i].y - 120)) > 6) && ((locbotg[i].y > 40) && (locbotg[i].y < 200))))
					{
						com.send_data(i + 1);
						Sleep(3);
						com.send_data(0x09);
						Sleep(3);
						com.send_data(180);
						Sleep(3);
						com.send_data(180);
						Sleep(3);
					}
					else
					{
						com.send_data(id + 1);
						Sleep(3);
						com.send_data(0);
						Sleep(3);
					}
				}
				cout << "Zooming OUT\n";
			}
			else
			{
				cout << "Cannot Zoom More!!!\n";
				com.send_data(0);
				Sleep(3);
				com.send_data(0);
				Sleep(3);
				com.send_data(0);
				Sleep(3);
				com.send_data(0);
				Sleep(3);
			}
		}
		else
		{
			cout << "Stopped\n";
			com.send_data(0);
			Sleep(3);
			com.send_data(0);
			Sleep(3);
			com.send_data(0);
			Sleep(3);
			com.send_data(0);
			Sleep(3);
		}
	}
	else
	{
		cout << "Stopped No Fingers\n";
		com.send_data(0);
		Sleep(3);
		com.send_data(0);
		Sleep(3);
		com.send_data(0);
		Sleep(3);
		com.send_data(0);
		Sleep(3);
	}
	imshow("Stream", img2);
}

void move_group()
{

}

void choose_controlled_bot()
{
	char robo_num[10];
	cam.read(img);
	//img = Mat::zeros(Size(320, 240), CV_8UC3);
	for (int i = 0; i < n_bots; i++)
	{
		circle(img, locbotg[i], 17, Scalar(0, 255, 0),2);
		_snprintf_s(robo_num, 10, "%d", i+1);
		putText(img, robo_num, Point(locbotg[i].x - 10, locbotg[i].y + 10), CV_FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255),2);
	}
	imshow("Stream", img);
	waitKey(100);
	prompt_id: clear_screen();
	cout << "Select Robot to be Controlled: ";
	controlled_bot_id = get_user_choice();
	if (controlled_bot_id > n_bots)
	{
		clear_screen();
		cout << "There are only " << n_bots << " robots available. " << endl;
		cout << "Please provide a valid input";
		Sleep(1000);
		goto prompt_id;
	}
	else
	{
		destroyWindow("Finger Blob");
		destroyWindow("Webcam Stream");
		controlled_bot_id -= 1;
	}
}

int bot_side_goal_dist(int g_no, int id)
{
	return (((locbotg[id].x - side_goals[g_no].x) * (locbotg[id].x - side_goals[g_no].x)) + ((locbotg[id].y - side_goals[g_no].y) * (locbotg[id].y - side_goals[g_no].y)));
}

void assign_goals_side()
{
	cam.read(img);
	cam.read(img);
	img2 = img.clone();
	for (int i = 0; i < n_bots; i++)
	{
		circle(img, side_goals[i], 3, Scalar(0, 255, 255), 1);
	}
	imshow("Stream", img);
	waitKey(10);
	int bot_goal_distances[100][100] = { 0 };
	int min_dist;
	int min_id;
	for (int i = 0; i < n_bots; i++)		//bot
	{
		for (int j = 0; j < n_bots; j++)	//goal
		{
			bot_goal_distances[i][j] = bot_side_goal_dist(j, i);
		}
		taken[i] = 0;
	}
	goals[controlled_bot_id] = side_goals[controlled_bot_id];
	taken[controlled_bot_id] = 1;

	int tot_dist[100] = { 0 };
	int id_sort[100] = { 0 };
	for (int i = 0; i < n_bots; i++)
	{
		for (int j = 0; j < n_bots; j++)
		{
			tot_dist[i] += bot_goal_distances[j][i];
		}
		id_sort[i] = i;
	}

	//sorting

	for (int i = 0; i < n_bots; i++)
	{
		for (int j = 0; j < n_bots - 1; j++)
		{
			if (tot_dist[j] < tot_dist[j + 1])
			{
				int t = tot_dist[j];
				tot_dist[j] = tot_dist[j + 1];
				tot_dist[j + 1] = t;
				t = id_sort[j];
				id_sort[j] = id_sort[j + 1];
				id_sort[j + 1] = t;
			}
		}
	}

	/*for (int i = 0; i < n_bots; i++)
	{
		cout << "Dist for goal: " << i + 1 << " = " << tot_dist[i] << endl;
	}
	waitKey(0);*/

	Scalar color;
	for (int i = 0; i < n_bots; i++)
	{
		if (id_sort[i] != controlled_bot_id)
		{
			img = img2.clone();
			min_dist = tot_dist[0];
			min_id = 0;
			if (i == 0)
			{
				color = Scalar(255, 0, 255);
			}
			else if (i == 1)
			{
				color = Scalar(0, 255, 0);
			}
			else if (i == 2)
			{
				color = Scalar(0, 0, 255);
			}
			if (i == 3)
			{
				color = Scalar(0, 255, 255);
			}
			if (i == 4)
			{
				color = Scalar(255, 255, 0);
			}
			for (int j = 0; j < n_bots; j++)
			{
				line(img, side_goals[id_sort[i]], locbotg[j], color, 1);
				if ((bot_side_goal_dist(id_sort[i], j) < min_dist) && (taken[j] == 0) && (j != controlled_bot_id))
				{
					min_dist = bot_side_goal_dist(id_sort[i], j);
					min_id = j;
				}
			}
			imshow("Stream", img);
			waitKey(300);
			goals[min_id] = side_goals[id_sort[i]];
			taken[min_id] = 1;
		}
	}
	img = img2.clone();
	for (int i = 0; i < n_bots; i++)
	{
		int min_dist = tot_dist[0];
		int min_id = 0;
		if (i == 0)
		{
			color = Scalar(255, 0, 255);
		}
		else if (i == 1)
		{
			color = Scalar(0, 255, 0);
		}
		else if (i == 2)
		{
			color = Scalar(0, 0, 255);
		}
		if (i == 3)
		{
			color = Scalar(0, 255, 255);
		}
		if (i == 4)
		{
			color = Scalar(255, 255, 0);
		}
		line(img, goals[i], locbotg[i], color, 1);
	}
	imshow("Stream", img);
	waitKey(10);
	Sleep(2000);
}

void clear_controlled_robot_path()
{
	cam.read(img);
	//img = Mat::zeros(Size(320, 240), CV_8UC3);
	int left_div = (n_bots - 1) / 2, left_count = 0;
	int right_div = n_bots / 2, right_count = 0;
	int div_count = 0;
	int i = 0;
	while(left_count < left_div)
	{
		int x_div = 40;
		int y_div;
		if (i > controlled_bot_id)
		{
			if (left_div > 1)
			{
				y_div = 40 + (160 / (left_div - 1))*(i - 1);
			}
			else
			{
				y_div = 40 + 160*(i - 1);
			}
		}
		else
		{
			if (left_div > 1)
			{
				y_div = 40 + (160 / (left_div - 1))*i;
			}
			else
			{
				y_div = 40 + (160)*i;
			}
		}
		if (i != controlled_bot_id)
		{
			side_goals[div_count] = Point(x_div, y_div);
			//cout << "Goal " << div_count + 1 << ": " << side_goals[div_count] << endl;
			circle(img, side_goals[div_count], 3, Scalar(255, 255, 0));
			left_count += 1;
			div_count += 1;
		}
		else
		{
			if (locbotg[controlled_bot_id].x <= 120)
			{
				side_goals[div_count] = Point(120, locbotg[controlled_bot_id].y);
			}
			else if (locbotg[controlled_bot_id].x >= 200)
			{
				side_goals[div_count] = Point(200, locbotg[controlled_bot_id].y);
			}
			else
			{
				side_goals[div_count] = locbotg[controlled_bot_id];
			}
			//cout << "Goal " << div_count + 1 << ": " << side_goals[div_count] << endl;
			circle(img, side_goals[div_count], 3, Scalar(255, 255, 0));
			div_count += 1;
		}
		i += 1;
	}
	imshow("Stream", img);
	waitKey(10);
	i = 0;
	while (right_count < right_div)
	{
		int x_div = 280;
		int y_div;
		if (i+left_div > controlled_bot_id)
		{
			if (controlled_bot_id < left_div)
			{
				if (right_div > 1)
				{
					y_div = 40 + (160 / (right_div - 1))*(i);
				}
				else
				{
					y_div = 40 + (160)*(i);
				}
			}
			else
			{
				if (right_div > 1)
				{
					y_div = 40 + (160 / (right_div - 1))*(i - 1);
				}
				else
				{
					y_div = 40 + (160)*(i - 1);
				}
			}
			
		}
		else
		{
			if (right_div > 1)
			{
				y_div = 40 + (160 / (right_div - 1))*(i);
			}
			else
			{
				y_div = 40 + (160)*(i);
			}
		}
		if (i+left_div != controlled_bot_id)
		{
			side_goals[div_count] = Point(x_div, y_div);
			//cout << "Goal " << div_count + 1 << ": " << side_goals[div_count] << endl;
			circle(img, side_goals[div_count], 3, Scalar(255, 255, 0));
			right_count += 1;
			div_count += 1;
		}
		else
		{
			if (locbotg[controlled_bot_id].x <= 120)
			{
				side_goals[div_count] = Point(120, locbotg[controlled_bot_id].y);
			}
			else if (locbotg[controlled_bot_id].x >= 200)
			{
				side_goals[div_count] = Point(200, locbotg[controlled_bot_id].y);
			}
			else
			{
				side_goals[div_count] = locbotg[controlled_bot_id];
			}
			//cout << "Goal " << div_count + 1 << ": " << side_goals[div_count] << endl;
			circle(img, side_goals[div_count], 3, Scalar(255, 255, 0));
			div_count += 1;
		}
		i += 1;
	}
	if (controlled_bot_id == n_bots - 1)
	{
		if (locbotg[controlled_bot_id].x >= 200)
		{
			side_goals[div_count] = Point(200, locbotg[controlled_bot_id].y);
		}
		else
		{
			side_goals[div_count] = locbotg[controlled_bot_id];
		}
		//cout << "Goal " << div_count + 1 << ": " << side_goals[div_count] << endl;
		circle(img, side_goals[div_count], 3, Scalar(255, 255, 0));
	}
	/*for (int i = 0; i < n_bots; i++)
	{
		cout << endl << side_goals[i];
	}*/

	imshow("Stream", img);
	waitKey(10);
	Sleep(1000);
	assign_goals_side();
	form_shape();
}

double point_distance(Point p1, Point p2)
{
	return (((p1.x - p2.x)*(p1.x - p2.x)) + ((p1.y - p2.y)*(p1.y - p2.y)));
}

void draw_robot_path()
{
	int counter = 0;
	cam.read(img);
	char robo_num[10];
	for (int i = 0; i < n_bots; i++)
	{
		circle(img, locbotg[i], 17, Scalar(0, 255, 0), 2);
		_snprintf_s(robo_num, 10, "%d", i + 1);
		putText(img, robo_num, Point(locbotg[i].x - 10, locbotg[i].y + 10), CV_FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
	}
	int draw_flag = 0, exit;
	drawing1 = Mat::zeros(img.size(), CV_8UC3);
	drawing2 = Mat::zeros(img.size(), CV_8UC3);
	drawing3 = Mat::zeros(img.size(), CV_8UC3);
	rectangle(drawing3, Point(60, 25), Point(260, 215), Scalar(255, 255), 3);
	drawing4 = Mat::zeros(img.size(), CV_8UC3);
	new_c = Point(-1, -1);
	old_c = Point(-1, -1);
	Sleep(2000);
	namedWindow("Finger Blob", CV_WINDOW_AUTOSIZE);
	while (1)
	{
		drawing1 = Mat::zeros(img.size(), CV_8UC3);
		webcam.read(img2);
		cvtColor(img2, img_hsv, CV_BGR2HSV);
		inRange(img_hsv, Scalar(minHFing, minSFing, minVFing), Scalar(maxHFing, maxSFing, maxVFing), thresh_finger);
		erode(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		dilate(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		flip(thresh_finger, thresh_finger, 1);
		Moments mom = moments(thresh_finger);
		if (mom.m00 > size)
		{
			new_c = Point((mom.m10 / mom.m00), (mom.m01 / mom.m00));
			cnew_c = Point((320 - new_c.x), new_c.y);
			if (old_c.x > -1)
			{
				if (point_distance(new_c, locbotg[controlled_bot_id]) < 225)
				{
					if (draw_flag == 0)
					{
						draw_flag = 2;
					}
				}
				else if (point_distance(new_c, locbotg[controlled_bot_id]) > 400)
				{
					if (draw_flag == 2)
					{
						draw_flag = 1;
					}
				}
				if (draw_flag == 0)
				{
					circle(drawing1, new_c, 3, Scalar(0, 0, 255), -1);
				}
				else
				{
					circle(drawing1, new_c, 3, Scalar(0, 255, 0), -1);
					if (draw_flag == 1)
					{
						if (new_c.x > 60 && new_c.x < 260 && new_c.y > 25 && new_c.y < 215)
						{
							line(drawing2, new_c, old_c, Scalar(150, 150, 0), 2);
							path.push_back(new_c);
						}
						if (point_distance(new_c, old_c) < 25)
						{
							counter += 1;
							//cout << counter << endl;
						}
						else
						{
							counter = 0;
						}
					}
				}
			}
			old_c = new_c;
		}
		else
		{
			old_c = Point(-1, -1);
		}
		//flip(drawing1, drawing3, 1);
		//flip(drawing2, drawing4, 1);
		//flip(img2, img2, 1);
		//flip(thresh_finger, thresh_finger, 1);
		imshow("Stream", img + drawing1 + drawing2 + drawing3);
		imshow("Finger Blob", thresh_finger);
		//imshow("YY", drawing4);
		//imshow("XX", drawing1);
		exit = waitKey(10);
		if (exit == 32 || exit == 27 || counter == 15)
		{
			if (exit == 27)
			{
				exit_flag = 1;
			}
			break;
		}
		if (exit == 'c')
		{
			old_c = Point(-1, -1);
			new_c = Point(-1, -1);
			//cold_c = Point(-1, -1);
			//cnew_c = Point(-1, -1);
			drawing2 = Mat::zeros(img.size(), CV_8UC3);
			//drawing1 = Mat::zeros(img.size(), CV_8UC3);
			path.clear();
			draw_flag = 0;
		}
	}
	destroyWindow("Finger Blob");
}

void go_to_goal_ind_nonstop(int id)
{
	get_xy_errors(id);
	int omega = (24 * e_phi) + (1 * e_phi_sum[id]);
	float x = (20 * e_x_bots) + (1 * e_x_sum[id]);// +(0.01*(e_x - ex_old));
	float y = (20 * e_y_bots) + (1 * e_y_sum[id]);// + (0.01*(e_y - ey_old));
	vel = sqrt((x*x) + (y*y));
	unsigned char vl = (unsigned char)((2 * vel - 15.5*omega) / 5);
	unsigned char vr = (unsigned char)((2 * vel + 15.5*omega) / 5);
	if (vr > 220)
	{
		vr = 220;
	}
	else if (vr < 170)
	{
		vr = 170;
	}
	if (vl > 220)
	{
		vl = 220;
	}
	else if (vl < 170)
	{
		vl = 170;
	}
	unsigned char motion;
	if (e_phi > 30)
	{
		motion = 0x0a;
	}
	else if (e_phi < -30)
	{
		motion = 0x05;
	}
	else if (e_phi > -30 && e_phi < -10)
	{
		motion = 0x04;
	}
	else if (e_phi > 10 && e_phi < 30)
	{
		motion = 0x02;
	}
	else
	{
		motion = 0x06;
	}
	if (((e_x_bots < 5 && e_x_bots > -5) && (e_y_bots < 5 && e_y_bots > -5)))
	{
		reached |= (0x01 << id);
		//motion = 0x00;
	}
	else
	{
		reached &= ~(0x01 << id);
	}
	if (motion != 0)
	{
		com.send_data(id + 1);
		Sleep(2);
		com.send_data(motion);
		Sleep(2);
		com.send_data(vl);
		Sleep(2);
		com.send_data(vr);
		Sleep(2);
	}
	else
	{
		com.send_data(id + 1);
		Sleep(2);
		com.send_data(motion);
		Sleep(2);
	}
}

void form_shape_ind_nonstop(int id)
{
	locate_bots_quick();
	get_angles();
	check_env();
	for (int i = 0; i < n_bots; i++)
	{
		if (i == controlled_bot_id)
		{
			get_goal_angle(id);
			go_to_goal_ind_nonstop(id);
		}
		else
		{
			com.send_data(i + 1);
			Sleep(5);
			com.send_data(0);
			Sleep(5);
		}
	}
	cout << endl;
}

void form_shape_ind(int id)
{
	locate_bots_quick();
	get_angles();
	check_env();
	for (int i = 0; i < n_bots; i++)
	{
		if (i == controlled_bot_id)
		{
			get_goal_angle(id);
			go_to_goal_ind(id);
		}
		else
		{
			com.send_data(i + 1);
			Sleep(5);
			com.send_data(0);
			Sleep(5);
		}
	}
	cout << endl;
}

void form_shape_ind_nonstop_no_obst(int id)
{
	locate_bots_quick();
	get_angles();
	check_env();
	for (int i = 0; i < n_bots; i++)
	{
		if (i == controlled_bot_id)
		{
			get_goal_angle(id);
			go_to_goal_ind(id);
		}
		else
		{
			com.send_data(i + 1);
			Sleep(5);
			com.send_data(0x00);
			Sleep(5);
		}
	}
	//avoid_obstacle_ind(id);
	//cout << endl;
}

void follow_path_robot(int bot_id)
{
	reached = 0;
	int exit;
	int reached_all = (0x01 << bot_id);
	goals[bot_id] = path[0];
	while (reached != reached_all)
	{
		form_shape_ind_nonstop(bot_id);
		imshow("Stream", img + drawing2);
		exit = waitKey(10);
		if (exit == 27)
		{
			reached = reached_all;
			exit_flag = 1;
		}
	}
	for (int i = 1; i < path.size(); i++)
	{
		if (point_distance(goals[bot_id], path[i]) > 25)
		{
			goals[bot_id] = path[i];
			reached = 0;
			while (reached != reached_all)
			{
				if (exit == 27)
				{
					reached = reached_all;
					exit_flag = 1;
				}
				form_shape_ind_nonstop(bot_id);
				imshow("Stream", img + drawing2);
				exit = waitKey(10);
			}
		}
	}
	goals[bot_id] = path[path.size() - 1];
	reached = 0;
	while (reached != reached_all)
	{
		if (exit == 27)
		{
			reached = reached_all;
			exit_flag = 1;
		}
		form_shape_ind(bot_id);
		imshow("Stream", img + drawing2);
		exit = waitKey(10);
	}
	com.send_data(0);
	Sleep(5);
	com.send_data(0);
	Sleep(5);
	com.send_data(0);
	Sleep(5);
	com.send_data(0);
	Sleep(5);
	com.send_data(0);
	Sleep(5);
	com.send_data(0);
	Sleep(5);
	com.send_data(0);
	Sleep(5);
	com.send_data(0);
	Sleep(5);
	path.clear();
}

void point_and_move(int cont_id)
{
	int move_flag = 0, exit;
	reached = 0;
	int reached_all = 0x01 << cont_id;
	drawing1 = Mat::zeros(img.size(), CV_8UC3);
	drawing2 = Mat::zeros(img.size(), CV_8UC3);
	drawing3 = Mat::zeros(img.size(), CV_8UC3);
	drawing4 = Mat::zeros(img.size(), CV_8UC3);
	new_c = Point(-1, -1);
	old_c = Point(-1, -1);
	Sleep(2000);
	namedWindow("Finger Blob", CV_WINDOW_AUTOSIZE);
	while (1)
	{
		drawing1 = Mat::zeros(img.size(), CV_8UC3);
		webcam.read(img2);
		cam.read(img);
		cvtColor(img2, img_hsv, CV_BGR2HSV);
		inRange(img_hsv, Scalar(minHFing, minSFing, minVFing), Scalar(maxHFing, maxSFing, maxVFing), thresh_finger);
		erode(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		dilate(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		flip(thresh_finger, thresh_finger, 1);
		Moments mom = moments(thresh_finger);
		if (mom.m00 > size)
		{
			move_flag = 1;
			new_c = Point((mom.m10 / mom.m00), (mom.m01 / mom.m00));
			if (old_c.x > -1)
			{
				circle(drawing1, new_c, 3, Scalar(0, 255, 0), -1);
				//line(drawing2, new_c, old_c, Scalar(0, 80, 120), 2);
			}
			old_c = new_c;
			goals[cont_id] = new_c;
		}
		else
		{
			old_c = Point(-1, -1);
			goals[cont_id] = locbotg[cont_id];
		}
		if (move_flag == 1)
		{
			form_shape_ind_nonstop_no_obst(cont_id);
			if (old_c.x == -1)
			{
				reached = 0;
			}
		}
		//form_shape_ind(id);
		//flip(drawing1, drawing3, 1);
		//flip(drawing2, drawing4, 1);
		//flip(img2, img2, 1);
		//flip(thresh_finger, thresh_finger, 1);
		imshow("Stream", img + drawing1);
		imshow("Finger Blob", thresh_finger);
		//imshow("YY", drawing4);
		//imshow("XX", drawing1);
		exit = waitKey(10);
		if (exit == 32 || exit == 27 || reached == reached_all)
		{
			if (exit == 27)
			{
				exit_flag = 1;
			}
			break;
		}
		if (exit == 'c')
		{
			old_c = Point(-1, -1);
			new_c = Point(-1, -1);
			//cold_c = Point(-1, -1);
			//cnew_c = Point(-1, -1);
			//drawing2 = Mat::zeros(img.size(), CV_8UC3);
			drawing1 = Mat::zeros(img.size(), CV_8UC3);
		}
	}
	destroyWindow("Finger Blob");
}

void point_and_assault()
{
	int move_flag = 0, exit;
	reached = 0;
	int reached_all = 0;
	for (int i = 0; i < n_bots; i++)
	{
		reached_all |= 0x01 << i;
	}
	int counter1 = 0, counter2 = 0;
	drawing1 = Mat::zeros(img.size(), CV_8UC3);
	drawing2 = Mat::zeros(img.size(), CV_8UC3);
	drawing3 = Mat::zeros(img.size(), CV_8UC3);
	drawing4 = Mat::zeros(img.size(), CV_8UC3);
	new_c = Point(-1, -1);
	Sleep(2000);
	namedWindow("Finger Blob", CV_WINDOW_AUTOSIZE);
	while (1)
	{
		drawing1 = Mat::zeros(img.size(), CV_8UC3);
		webcam.read(img2);
		cam.read(img);
		cvtColor(img2, img_hsv, CV_BGR2HSV);
		inRange(img_hsv, Scalar(minHFing, minSFing, minVFing), Scalar(maxHFing, maxSFing, maxVFing), thresh_finger);
		erode(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		dilate(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		dilate(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
		erode(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
		flip(thresh_finger, thresh_finger, 1);
		Moments mom = moments(thresh_finger);
		if (mom.m00 > size)
		{
			new_c = Point((mom.m10 / mom.m00), (mom.m01 / mom.m00));
			if (counter1 < 10)
			{
				counter1 += 1;
				circle(drawing1, new_c, 3, Scalar(0, 0, 255), -1);
			}
			else
			{
				circle(drawing1, new_c, 3, Scalar(0, 255, 0), -1);
				//line(drawing2, new_c, old_c, Scalar(0, 80, 120), 2);
				locate_bots_quick();
				get_angles();
				for (int i = 0; i < n_bots; i++)
				{
					goals[i] = new_c;
					correct_angle_ind(i);
				}
				if (point_distance(new_c, old_c) < 49)
				{
					if (reached == reached_all)
					{
						counter2 += 1;
					}
				}
				else
				{
					counter2 = 0;
				}
			}	
			cout << counter1 << " " << counter2 << endl;
			old_c = new_c;
		}
		else
		{
			old_c = Point(-1, -1);
			counter2 = 0;
			move_flag = 0;
			com.send_data(0);
			Sleep(5);
			com.send_data(0);
			Sleep(5);
			com.send_data(0);
			Sleep(5);
			com.send_data(0);
			Sleep(5);
			com.send_data(0);
			Sleep(5);
			cout << endl;
		}
		//form_shape_ind(id);
		//flip(drawing1, drawing3, 1);
		//flip(drawing2, drawing4, 1);
		//flip(img2, img2, 1);
		//flip(thresh_finger, thresh_finger, 1);
		imshow("Stream", img + drawing1);
		imshow("Finger Blob", thresh_finger);
		//imshow("YY", drawing4);
		//imshow("XX", drawing1);
		exit = waitKey(10);
		if (exit == 32 || exit == 27 || counter2 >= 15)
		{
			if (exit == 27)
			{
				exit_flag = 1;
			}
			break;
		}
		/*if (exit == 'c')
		{
			old_c = Point(-1, -1);
			new_c = Point(-1, -1);
			//cold_c = Point(-1, -1);
			//cnew_c = Point(-1, -1);
			//drawing2 = Mat::zeros(img.size(), CV_8UC3);
			drawing1 = Mat::zeros(img.size(), CV_8UC3);
		}*/
	}
	clear_screen();
	destroyWindow("Finger Blob");
}

void pinch_and_zoom()
{
	//namedWindow("Finger Stream", CV_WINDOW_AUTOSIZE);
	int exit = 0;
	while (1)
	{
		locate_zoom_markers();
		zoom();
		exit = waitKey(10);
		if (exit == 27)
		{
			exit_flag = 1;
			break;
		}
	}
}

void draw_and_save(int num)
{
	if (num > 0)
	{
		flagfile.open("C:/Users/ABHISHEK/Desktop/Abhishek Stuff/Robotics/Online Courses/Machine Learning/Programming Exercises/Week 5/mlclass-ex4-006/mlclass-ex4/BE Project/flag.txt");
		flagfile << "0";
		flagfile.close();
		int y = num;
		number_egfile.open("C:/Users/ABHISHEK/Desktop/Abhishek Stuff/Robotics/Online Courses/Machine Learning/Programming Exercises/Week 5/mlclass-ex4-006/mlclass-ex4/BE Project/num.txt");
		number_egfile << y;
		number_egfile.close();
		mode_file.open("C:/Users/ABHISHEK/Desktop/Abhishek Stuff/Robotics/Online Courses/Machine Learning/Programming Exercises/Week 5/mlclass-ex4-006/mlclass-ex4/BE Project/MatOCV Comm/mode.txt");
		mode_file << "1";
		mode_file.close();
		cout << "Enter Shape Index: \n1 = V\n2 = Rectangle\n3 = Circle\n4 = T\n5 = Inverted V\n6 = Line\nEnter choice: ";
		cin >> y;
		opfile.open("C:/Users/ABHISHEK/Desktop/Abhishek Stuff/Robotics/Online Courses/Machine Learning/Programming Exercises/Week 5/mlclass-ex4-006/mlclass-ex4/BE Project/output.txt");
		opfile << y;
		opfile.close();
		//cout << "\n\nSTART MATLAB acquire_dataset NOW!!!\n";
		webcam.read(img);
		Mat drawing1 = Mat::zeros(img.size(), CV_8UC1);
		Mat drawing2 = Mat::zeros(img.size(), CV_8UC1);
		Mat drawing3 = Mat::zeros(Size(20, 20), CV_8UC1);
		Mat drawing4 = Mat::zeros(Size(20, 20), CV_8UC1);
		namedWindow("XX", CV_WINDOW_AUTOSIZE);
		namedWindow("YY", CV_WINDOW_AUTOSIZE);
		for (int i = 0; i < num; i++)
		{
			clear_screen();
			old_c = Point(-1, -1);
			new_c = Point(-1, -1);
			cold_c = Point(-1, -1);
			cnew_c = Point(-1, -1);
			drawing2 = Mat::zeros(img.size(), CV_8UC1);
			drawing3 = Mat::zeros(Size(20, 20), CV_8UC1);
			cout << "Draw Training Shapes here. \n\nNOTE: Press SPACE when done drawing. \n      Press and hold 'c' to clear drawing and redraw. \n\n";
			cout << "Draw Training Shape No. " << i + 1 << ": " << endl;
			//system("pause");

			webcam.read(img);
			webcam.read(img);
			while (1)
			{
				webcam.read(img);
				cvtColor(img, img_hsv, CV_BGR2HSV);
				inRange(img_hsv, Scalar(minHFing, minSFing, minVFing), Scalar(maxHFing, maxSFing, maxVFing), thresh_finger);
				erode(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
				dilate(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
				dilate(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
				erode(thresh_finger, thresh_finger, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
				Moments mom = moments(thresh_finger);
				if (mom.m00 > size)
				{
					new_c = Point((mom.m10 / mom.m00), (mom.m01 / mom.m00));
					cnew_c = Point((mom.m10 / (16 * mom.m00)), (mom.m01 / (12 * mom.m00)));
					if (old_c.x > -1)
					{
						line(drawing2, new_c, old_c, Scalar(255), 3);
						line(drawing3, cnew_c, cold_c, Scalar(255), 1);
					}
					old_c = new_c;
					cold_c = cnew_c;
				}
				flip(drawing2, drawing1, 1);
				flip(drawing3, drawing4, 1);
				flip(img, img, 1);
				flip(thresh_finger, thresh_finger, 1);
				imshow("Stream", img);
				imshow("Yellow Blob", drawing1);
				imshow("YY", drawing4);
				imshow("XX", thresh_yellow);
				if (waitKey(10) == 32)
				{
					break;
				}
				if (waitKey(10) == 'c')
				{
					old_c = Point(-1, -1);
					new_c = Point(-1, -1);
					cold_c = Point(-1, -1);
					cnew_c = Point(-1, -1);
					drawing2 = Mat::zeros(img.size(), CV_8UC1);
					drawing3 = Mat::zeros(Size(20, 20), CV_8UC1);
				}
			}
			//cout << "Please enter desired O/P: ";
			//cin >> y;
			//resize(drawing1, drawing1, Size(20, 20));
			imwrite("C:/Users/ABHISHEK/Desktop/Abhishek Stuff/Robotics/Online Courses/Machine Learning/Programming Exercises/Week 5/mlclass-ex4-006/mlclass-ex4/BE Project/training_eg.jpg", drawing4);
			/*opfile.open("C:/Users/ABHISHEK/Desktop/Abhishek Stuff/Robotics/Online Courses/Machine Learning/Programming Exercises/Week 5/mlclass-ex4-006/mlclass-ex4/BE Project/output.txt");
			opfile << y;
			opfile.close();*/
			flagfile.open("C:/Users/ABHISHEK/Desktop/Abhishek Stuff/Robotics/Online Courses/Machine Learning/Programming Exercises/Week 5/mlclass-ex4-006/mlclass-ex4/BE Project/flag.txt");
			flagfile << "1";
			flagfile.close();
		}
		clear_screen();
		cout << "Please verify your Training Examples in the Matlab Window.\n";
		system("pause");
	}
}

Mat hwnd2mat(HWND hwnd){

	HDC hwindowDC, hwindowCompatibleDC;

	int height, width, srcheight, srcwidth;
	HBITMAP hbwindow;
	Mat src;
	BITMAPINFOHEADER  bi;

	hwindowDC = GetDC(hwnd);
	hwindowCompatibleDC = CreateCompatibleDC(hwindowDC);
	SetStretchBltMode(hwindowCompatibleDC, COLORONCOLOR);

	RECT windowsize;    // get the height and width of the screen
	GetClientRect(hwnd, &windowsize);

	srcheight = windowsize.bottom;
	srcwidth = windowsize.right;
	height = windowsize.bottom;  //change this to whatever size you want to resize to
	width = windowsize.right;

	src.create(height, width, CV_8UC4);

	// create a bitmap
	hbwindow = CreateCompatibleBitmap(hwindowDC, width, height);
	bi.biSize = sizeof(BITMAPINFOHEADER);    //http://msdn.microsoft.com/en-us/library/windows/window/dd183402%28v=vs.85%29.aspx
	bi.biWidth = width;
	bi.biHeight = -height;  //this is the line that makes it draw upside down or not
	bi.biPlanes = 1;
	bi.biBitCount = 32;
	bi.biCompression = BI_RGB;
	bi.biSizeImage = 0;
	bi.biXPelsPerMeter = 0;
	bi.biYPelsPerMeter = 0;
	bi.biClrUsed = 0;
	bi.biClrImportant = 0;

	// use the previously created device context with the bitmap
	SelectObject(hwindowCompatibleDC, hbwindow);
	// copy from the window device context to the bitmap device context
	StretchBlt(hwindowCompatibleDC, 0, 0, width, height, hwindowDC, 0, 0, srcwidth, srcheight, SRCCOPY); //change SRCCOPY to NOTSRCCOPY for wacky colors !
	GetDIBits(hwindowCompatibleDC, hbwindow, 0, height, src.data, (BITMAPINFO *)&bi, DIB_RGB_COLORS);  //copy from hwindowCompatibleDC to hbwindow

	// avoid memory leak
	DeleteObject(hbwindow); DeleteDC(hwindowCompatibleDC); ReleaseDC(hwnd, hwindowDC);

	return src;
}

int main()
{

	init_database();
	init_shapes();
	init_shape_names();
	mode_file.open("C:/Users/ABHISHEK/Desktop/Abhishek Stuff/Robotics/Online Courses/Machine Learning/Programming Exercises/Week 5/mlclass-ex4-006/mlclass-ex4/BE Project/MatOCV Comm/mode.txt");
	mode_file << "0"; 
	mode_file.close();
	exit_file.open("C:/Users/ABHISHEK/Desktop/Abhishek Stuff/Robotics/Online Courses/Machine Learning/Programming Exercises/Week 5/mlclass-ex4-006/mlclass-ex4/BE Project/MatOCV Comm/exit.txt");
	exit_file << "0";
	exit_file.close();
	while (mode == 0)
	{
		start_file.open("C:/Users/ABHISHEK/Desktop/Abhishek Stuff/Robotics/Online Courses/Machine Learning/Programming Exercises/Week 5/mlclass-ex4-006/mlclass-ex4/BE Project/MatOCV Comm/start.txt");
		start_file >> mode;
		start_file.close();
		if (mode == 0)
		{
			clear_screen();
			cout << "Error!!! Matlab code \"Gesture_recognition.m\" is not running. \nPlease run the code and then ";
			Sleep(100);
		}
	}
	
	//initialise_system();
	calibrate_gesture_markers();
	calibrate_all_finger_markers();
	do
	{
		destroyWindow("Finger Tracking");
	mode_select: clear_screen();
		cout << "Select Mode: \n1. Train Gesture Recognition System. \n2. Run Gesture Controlled Multi-Robotic System \n3. Exit \nEnter your choice: ";
		mode = get_user_choice();
		/*if ((m >= 0x30) && (m <= 0x39))
		{
			mode = (int)(m & 0x0f);
		}
		else
		{
			clear_screen();
			cout << "Please Enter a valid choice...\n\n";
			goto mode_select;
		}*/
		if (mode > 3)
		{
			clear_screen();
			cout << "Please Enter a valid choice...\n\n";
			Sleep(1000);
			goto mode_select;
		}
		destroyWindow("Finger Blob");
		destroyWindow("Webcam Stream");
		clear_screen();
		if (mode == 1)
		{
			cout << "You selected TRAINING MODE. \n\n";
			int numbers;
			cout << "\n\nEnter Training Session Length (*No. of Training Examples in multiples of 10*): \n";
			numbers = (get_user_choice() * 10);
			destroyWindow("Finger Blob");
			destroyWindow("Webcam Stream");
			draw_and_save(numbers);
		}
		else if (mode == 2)
		{
			mode_file.open("C:/Users/ABHISHEK/Desktop/Abhishek Stuff/Robotics/Online Courses/Machine Learning/Programming Exercises/Week 5/mlclass-ex4-006/mlclass-ex4/BE Project/MatOCV Comm/mode.txt");
			mode_file << "2";
			mode_file.close();
			cout << "You selected GESTURE CONTROLLED MULTI-ROBOTICS SYSTEM MODE\n\n";
			cout << "Checking XBee Transmitter Connection: ";
			com.startDevice("COM7", 9600);
			cout << "Checking Overhead Camera Connection: ";
			if (!cam.isOpened())
			{
				cout << "Camera Not Opened...\n\n";
				system("pause");
				return -1;
			}
			else
			{
				cout << "Camera Connected Successfully \n\n";
				init_camera_quad();
				init_webcam();
			}
			/*cout << "Enter the Number of Robots: ";
			cin >> n_bots;*/
			init_bots_state();
			clear_screen();
			if (calib_flag == 0)
			{
				clear_screen();
				cout << "Camera Calibration Phase\n\n";
				calibrate();
				calib_flag = 1;
			}
			if (count_flag == 0)
			{
				clear_screen();
				cout << "Counting Number of Robots in Arena\n\n";
				count_bots();
				count_flag = 1;
			}
			if (id_flag == 0)
			{
				clear_screen();
				cout << "Obtaining Mobile Robot IDs\n\n";
				Sleep(1000);
				locate_bots_temp();
				calib_id_quick();
				id_flag = 1;
			}
			clear_screen();
			cout << "Preparing for Human Interface and Control Phase.\n\n";
			Sleep(500);
			int ans;
		cal_prompt: cout << "Do you want to Calibrate Finger Markers?\n1.Yes\n2.No";
			ans = get_user_choice();
			if (ans == 1)
			{
				fing_calib_flag = 0;
			}
			else if (ans == 2)
			{
				fing_calib_flag = 1;
			}
			else
			{
				clear_screen();
				cout << "Please select valid input.";
				Sleep(1000);
				clear_screen();
				goto cal_prompt;
			}
			if (fing_calib_flag == 0)
			{
				clear_screen();
				calibrate_gesture_markers();
				destroyWindow("Finger Tracking");
			}
			Sleep(1000);
			//return 0;
			while (mode != 7)
			{
				prompt: clear_screen();
				cout << "Enter Robot Control Mode: \n1: Shape Formation \n2: Group Locomotion \n3: Point and Move \n4: Individual Path Follow \n5: Point and Assault \n6: Obtain Robot ID's \n7: Exit \n\nEnter your Choice: ";
				mode = get_user_choice();
				/*if ((m < 0x30) || (m > 0x39))
				{
					clear_screen();
					cout << "Enter a valid input." << endl;
					Sleep(1);
					goto prompt;
				}
				else
				{
					mode = (int)(m & 0x0f);
				}*/
				if (mode > 7)
				{
					clear_screen();
					cout << "Enter a valid input." << endl;
					Sleep(1000);
					goto prompt;
				}
				destroyWindow("Finger Blob");
				destroyWindow("Webcam Stream");
				exit_flag = 0;
				switch (mode)
				{
				case 1:	while(exit_flag == 0)
				{
				back: clear_screen();
					get_hand_out();
					cout << "Please Draw the Shape to be made by the robots. \n\nUSE FINGER TO DRAW IN THE AIR. \nYour Drawing will appear on the \"Drawing\" Window: \n\nPress SPACE to Accept. \nPress and Hold 'c' to Clear the Drawing. \n\nPress and Hold 'esc' to go to previous menu.";
					recognise_gesture();
					if (exit_flag == 0)
					{
						clear_screen();
						int ans;
					shape_q: cout << "Is this the Intended Shape? \n1. Yes\n2. No";
						ans = get_user_choice();
						if (ans == 2)
						{
							goto back;
						}
						else if (ans == 1)
						{

						}
						else
						{
							clear_screen();
							cout << "\nPlease enter a valid input...";
							Sleep(1000);
							clear_screen();
							goto shape_q;
						}
						destroyWindow("Finger Blob");
						destroyWindow("Webcam Stream");
						assign_goals();
						form_shape();
						if (exit_flag == 0)
						{
							int ans;
							clear_screen();
						prompt1: cout << "Do you want to form another shape? \n1. Yes \n2. No\n";
							ans = get_user_choice();
							if (ans == 2)
							{
								exit_flag = 1;
							}
							else if (ans == 1)
							{

							}
							else
							{
								clear_screen();
								cout << "Please enter a valid input...";
								Sleep(1000);
								clear_screen();
								goto prompt1;
							}
						}
					}
				}
					break;
				case 2: while (exit_flag == 0)
				{
					clear_screen();
					//save_current_locations();
					cout << "MOVE FORWARD: Move PALM Closer to camera. \nSTOP: Move PALM away from camera. \nROTATE LEFT: Move PALM to left side of screen. \nROTATE RIGHT: Move PALM to right side of the screen. \nEXIT: Move PALM out of screen. ";
					//move_group();
				}
					break;
				case 3: while (exit_flag == 0)
				{
					clear_screen();
					choose_controlled_bot();
					get_hand_out();
					point_and_move(controlled_bot_id);
					if (exit_flag == 0)
					{
						int ans;
						clear_screen();
						prompt3: cout << "Do you want to control more robots individually? \n1. Yes \n2. No\n";
						ans = get_user_choice();
						if (ans == 2)
						{
							exit_flag = 1;
						}
						else if (ans == 1)
						{

						}
						else
						{
							clear_screen();
							cout << "Please enter a valid input...";
							Sleep(1000);
							clear_screen();
							goto prompt3;
						}
						destroyWindow("Finger Blob");
						destroyWindow("Webcam Stream");
					}
				}
					break;
				case 4:	while (exit_flag == 0)
				{
					clear_screen();
					choose_controlled_bot();
					clear_controlled_robot_path();
					get_hand_out();
					draw_robot_path();
					follow_path_robot(controlled_bot_id);
					if (exit_flag == 0)
					{
						int ans;
						clear_screen();
					prompt4: cout << "Do you want to control more robots individually? \n1. Yes \n2. No\n";
						ans = get_user_choice();
						if (ans == 2)
						{
							exit_flag = 1;
						}
						else if (ans == 1)
						{

						}
						else
						{
							clear_screen();
							cout << "Please enter a valid input...";
							Sleep(1000);
							clear_screen();
							goto prompt4;
						}
						destroyWindow("Finger Blob");
						destroyWindow("Webcam Stream");
					}
				}
					break;
				case 5: while (exit_flag == 0)
				{
					clear_screen();
					get_hand_out();
					point_and_assault();
					if (exit_flag == 0)
					{
						int ans;
						clear_screen();
					prompt5: cout << "Do you want to try that again? \n1. Yes \n2. No\n";
						ans = get_user_choice();
						if (ans == 2)
						{
							exit_flag = 1;
						}
						else if (ans == 1)
						{

						}
						else
						{
							clear_screen();
							cout << "Please enter a valid input...";
							Sleep(1000);
							clear_screen();
							goto prompt5;
						}
						destroyWindow("Finger Blob");
						destroyWindow("Webcam Stream");
					}
				}
					break;
				case 6: clear_screen();
					cout << "Obtaining Mobile Robot IDs\n\n";
					Sleep(1000);
					locate_bots_temp();
					calib_id_quick();
					clear_screen();
					break;
				case 7: clear_screen();
					break;
				default: clear_screen();
					cout << "Please enter a valid Option.\n";
					Sleep(1000);
					break;
				}
			}
			mode = 0;
		}
		else if (mode == 3)
		{
			exit_file.open("C:/Users/ABHISHEK/Desktop/Abhishek Stuff/Robotics/Online Courses/Machine Learning/Programming Exercises/Week 5/mlclass-ex4-006/mlclass-ex4/BE Project/MatOCV Comm/exit.txt");
			exit_file << "1";
			exit_file.close();
			destroyWindow("Stream");
			cout << "Thank You.\n";
			Sleep(1000);
		}
		else
		{
			goto mode_select;
		}
	} while (mode != 3);
	return 0;
}