/**
	Author: Yeo Qin Jie Alvin
	Version: 1.5
	Date: 18th November 2007
	Last Updated: 23rd October 2007/10:30pm
	Last Updated: 30th October 2007/7:42pm

	4) 18th November - Edited Datapoint

	3) 30th October - Added Datapoint

	2) 23rd October - Added Face Detection and Pattern Recognition.

	1) 21st October - Includes all that is required to process the images.
*/

// declare header files
#include <cv.h>
#include <highgui.h>
#include <cxcore.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <iostream>
#include <fstream>

// Constants Required
#define TOLERANCE 25 // tolerance to detect a different pixel range  MIN 0 MAX 255
#define BLUE 255
#define GREEN 255
#define RED 255
#define MAXSIZE 800 // array max size
#define TOLERANCE_WHITE 255 // tolerance to detect a different pixel range  MIN 0 MAX 255
#define MODEL 400 // height of model
#define BLOCKSIZE 5
#define HEIGHT_MULTIPLIER 6

using namespace std; // to use vector

// enumeration
enum bodyLoc{
	TIP = 1, NECK, SHOULDER, HAND, ARMPIT, WAIST, PELVIS, BUFFER
};

enum objectPath { FACE = 1, PATTERN };

// casade for Face Detection
static const char* cascade_name_face = "data/haarcascade_frontalface_alt2.xml";

// casade for Pattern Recognition
static const char* cascade_name_pattern = "data/data.xml";

// create memory for Face Detection and Pattern Recognition
static CvMemStorage* storage = 0;
static CvHaarClassifierCascade* cascade = 0;

class ImgPro {
	

public:
	// Variables //////////////////////////////////
	// Functions //////////////////////////////////

	// Constructor
	
	ImgPro();

	// Destructor

	~ImgPro();

	static IplImage* cvtImage(IplImage*);
	
	// Image Differencing	
	static IplImage* absDiff(IplImage*, IplImage*);

	// Line Comparison, for 1 channel and 3 channels
	static int pixel1EqCheck(int, int, IplImage*);
	static bool pixel3EqCheck(int, int, IplImage*, IplImage*);
	
	// Pixel Comparison
	static bool pixelEqCheck(int , int );
	static bool isWhite(int, int, int, IplImage*);
	static bool is3White(int, int, IplImage*);

	// Get pixel information
	static uchar getData(int x, int y, int, IplImage*);

	// Face Detection
	static CvPoint detectFace(IplImage*, char*, int, int);

	// Pattern Recognition
	static CvSize getActualSize(IplImage*);

	// Data Point 
	static vector<int> dataPoint(IplImage*, CvSize, char*, char*, CvPoint);
};

// Definitions

// function to convert colour image to black and white image
IplImage* ImgPro::cvtImage(IplImage* image) {
	IplImage* blackImage = 0;

	blackImage = cvCreateImage(cvSize(image->width, image->height), image->depth, image->nChannels);
	
	for (int i = 0; i < image->height; i++) {
		for (int j = 0; j < image->width; j++) {
			if (!is3White(i, j, image)) {
				((uchar*)(blackImage->imageData + blackImage->widthStep*i))[j*3] = 0; 
				((uchar*)(blackImage->imageData + blackImage->widthStep*i))[j*3+1] = 0;
				((uchar*)(blackImage->imageData + blackImage->widthStep*i))[j*3+2] = 0;
			} else {
				((uchar*)(blackImage->imageData + blackImage->widthStep*i))[j*3] = BLUE; 
				((uchar*)(blackImage->imageData + blackImage->widthStep*i))[j*3+1] = GREEN;
				((uchar*)(blackImage->imageData + blackImage->widthStep*i))[j*3+2] = RED;
			}
		}
	}
	return blackImage;
}

IplImage* ImgPro::absDiff(IplImage* img1, IplImage* img2) {	
	IplImage* tempImage = 0;

	CvSize tempImgSize;
	tempImgSize.height = img1->height;
	tempImgSize.width = img1->width;

	tempImage = cvCreateImage(tempImgSize, img1->depth, img1->nChannels);

	for (int i = 0; i < tempImgSize.height; i++) {
		for (int j = 0; j <tempImgSize.width; j++) {
			if (pixel3EqCheck(i, j, img1, img2)) {
				// This is the background, can change to any colour based on the RGB colour
				// Another picture can be used as the background
				((uchar*)(tempImage->imageData + tempImage->widthStep*i))[j*3] = BLUE; 
				((uchar*)(tempImage->imageData + tempImage->widthStep*i))[j*3+1] = GREEN;
				((uchar*)(tempImage->imageData + tempImage->widthStep*i))[j*3+2] = RED;
			}
			else {
				// This is the original picture you wanted to extract out.
				((uchar*)(tempImage->imageData + tempImage->widthStep*i))[j*3] = ((uchar*)(img2->imageData + img2->widthStep*i))[j*3];
				((uchar*)(tempImage->imageData + tempImage->widthStep*i))[j*3+1] = ((uchar*)(img2->imageData + img2->widthStep*i))[j*3+1];
				((uchar*)(tempImage->imageData + tempImage->widthStep*i))[j*3+2] = ((uchar*)(img2->imageData + img2->widthStep*i))[j*3+2];
			}
		}
	}

	return tempImage;
};

int ImgPro::pixel1EqCheck(int x, int y, IplImage* img) {
	int imgPixel = getData(x, y, 0, img);

	return (imgPixel <= TOLERANCE) ? 1 : 0;
};

bool ImgPro::pixel3EqCheck(int x, int y, IplImage* img1, IplImage* img2) {
	unsigned int img1Pixel, img2Pixel;

	img1Pixel = getData(x, y, 0, img1);
	img2Pixel = getData(x, y, 0, img2);

	if (pixelEqCheck(img1Pixel, img2Pixel)) {
		img1Pixel = getData(x, y, 1, img1);
		img2Pixel = getData(x, y, 1, img2);
	} else return false;

	if (pixelEqCheck(img1Pixel, img2Pixel)) {
		img1Pixel = getData(x, y, 2, img1);
		img2Pixel = getData(x, y, 2, img2);
	} else return false;

	if (pixelEqCheck(img1Pixel, img2Pixel)) {
		return true;
	}

	return false;
};

bool ImgPro::pixelEqCheck(int pixel1, int pixel2) {
	int checkNum = pixel1 - pixel2;
	checkNum = (checkNum < 0) ? (checkNum * -1) : checkNum;

	return (checkNum <= TOLERANCE)? true : false;
}

bool ImgPro::isWhite(int x, int y, int offset, IplImage* img) {
	int imgPixel = getData(x, y, 0, img);

	return (imgPixel >= TOLERANCE_WHITE || imgPixel == -1) ? true : false;
}

bool ImgPro::is3White(int x, int y, IplImage* img1) {
	unsigned int img1Pixel;

	img1Pixel = getData(x, y, 0, img1);

	if (isWhite(x, y, 0, img1)) {
		img1Pixel = getData(x, y, 1, img1);
	} else return false;

	if (isWhite(x, y, 1, img1)) {
		img1Pixel = getData(x, y, 2, img1);
	} else return false;

	if (isWhite(x, y, 2, img1)) {
		return true;
	}

	return false;
};
uchar ImgPro::getData(int x, int y, int offset, IplImage* img) {
	return ((uchar*)(img->imageData + img->widthStep*x))[y*3 + offset];
}

CvPoint ImgPro::detectFace(IplImage* img, char* saveName, int offset, int path) {
    static CvScalar colors[] = {{0,0,255}};

    double scale = 1; 

	CvPoint cardLoc;

    if( cascade )
    {
        CvSeq* faces;

		// find faces/objects, depending on the cascade files

		try {
			faces = cvHaarDetectObjects( img, cascade, storage, 1.1, 2, 0, cvSize(30, 30) );
		} catch (...) {
			printf("Objection Detection Error!\n");
			exit(1);
		}

        CvRect* r = (CvRect*)cvGetSeqElem( faces, 0 );
        
		if (!r) {
			if (path == FACE)printf("Face Detection cannot be completed!\n");
			else printf("Pattern Detection cannot be completed!\n");
			exit(0);
		} 
		
		CvPoint center;
        int radius;
		// value of distance from center to edge of square
		int pT;
		CvPoint pt1, pt2;

		IplImage* imageReq;
		CvSize size1;
		CvMat imgMat;
		CvRect rect;
		
        center.x = cvRound((r->x + r->width*0.5)*scale);
        center.y = cvRound((r->y + r->height*0.5)*scale);
        radius = cvRound((r->width + r->height)*0.2*scale);

		cardLoc = center;

		pT = cvRound(sqrt((double)2*radius*radius));
		pt1.x = center.x - pT;
		pt1.y = center.y - pT - offset * 1.5;
		pt2.x = center.x + pT;
		pt2.y = center.y + pT + offset;

        //cvRectangle( img, pt1, pt2, colors[0], 1, 8, 0 );
			
		size1.height = pt2.y - pt1.y - 1;
		size1.width = pt2.x - pt1.x - 1;

		imgMat = cvMat(size1.height, size1.width, CV_32SC2, 0);
		rect = cvRect(pt1.x + 1, pt1.y + 1, size1.width, size1.height);
			
		imageReq = cvCreateImage(size1, 8, 1);

		// get the face
		cvGetSubRect(img, (CvMat*)imageReq, rect);
		cvSaveImage(saveName, imageReq);
    }

	return cardLoc;
}

CvSize ImgPro::getActualSize(IplImage * image) {
	CvSize sizeActual;
	int x, y;
	CvPoint pt1, pt2;
	int flag, flag2;
	IplImage* blackAndWhiteImage;
	CvSize size1;

	size1.height = image->height;
	size1.width = image->width;
	blackAndWhiteImage = cvCreateImage(size1,image->depth, 1);

	// change image to a gray scale image, eliminates colour noises
	cvCvtColor(image, blackAndWhiteImage, CV_BGR2GRAY);

	// flag to allow the code to run once only per execution
	flag = 1;

	// flag to test if array of pixel at y-axis has a black pixel
	flag2 = 1;

	for (x = 0; x < image->height; x++) {
		if (flag == 3 && flag2 == 0) {
			pt2.y = x;
			break;
		}

		flag2 = 0;

		for (y = 0; y < image->width; y++) {
			if (pixel1EqCheck(x, y, blackAndWhiteImage) && flag == 1) {
				
				// increase flag by 1
				flag++;
				// get coordinates for point 1
				pt1.x = y;
				pt1.y = x;
			} else

			if (flag == 2 && !pixel1EqCheck(x, y, blackAndWhiteImage)) {
				
				// increase flag by 1
				flag++;

				// get x-coordinate for point 2 
				pt2.x = y;
				flag2 = 1;
				break;
			} else

			if (flag == 3 && pixel1EqCheck(x, y, blackAndWhiteImage)) {
				flag2 = 1;
				break;
			}
		}
	}

	sizeActual.height = pt2.y - pt1.y;
	sizeActual.width = pt2.x - pt1.x;

	return sizeActual;
};

vector<int> ImgPro::dataPoint(IplImage * image, CvSize size, char* saveName, char* loadName, CvPoint cardLoc) {
	IplImage* blackAndWhiteImage = 0, *faceImage = 0;
	CvSize size1;
	ofstream dataFile;
	
	faceImage = cvLoadImage(loadName);
	const int faceHeight = faceImage->height;
	const int faceWidth = faceImage->width;

	// Create/Open text file for writing
	dataFile.open(saveName);

	const double widthI = 11.1; // Actual width of Pattern I in cm
	const double heightI = 22.1; // Actual height of Pattern I in cm
	
	// Get image height, sets local variable as there is multiple calling of references
	const int imageHeight = image->height;
	
	// Pattern Card actual size in centimetres for counting real measurements of user
	// Average of the two measurements to minimise errors
	double cmPerPixelWidth = widthI / (double)(size.width);
	double cmPerPixelHeight = heightI / (double)(size.height);
	double cmPerPixel = (cmPerPixelWidth + cmPerPixelHeight) / 2;

	// Allow the program to detect the critical points at specific areas. 
	int flag = 1, isReal; 

	 //Critical/Relative Points that are used in the programme to get datapoints
	 //Critical Points are points found on the actual Image
	 //Relative Points contains fixed points and points relative points computed from the critical points
	CvPoint critPts[9]; // critPts[0] not in use
	CvPoint relPts[8]; // relPts[0] not in use

	// Some set of standard points
	CvPoint standpts[] = {{0,0},{255,10},{209,76},{161,109},{138, 322}, {181, 144},{181,269},{225,269}};
	
	// Fixed Point
	relPts[TIP] = cvPoint(225, 10);

	// Size of actual image
	size1.height = imageHeight;
	size1.width = image->width;
	blackAndWhiteImage = cvCreateImage(size1,1, 1);

	// change image to a gray scale image, eliminates colour noises
	blackAndWhiteImage = cvtImage(image);

	// to save the data acquired by line scanning inside a vector data structure
	bool flag1 = true;
	bool write = false;
	vector<CvPoint> vecPts[MAXSIZE];
	int saveLocation = 0;

	// get all points location
	for (int i = 0; i < imageHeight; i++) {
		flag1 = true;
		write = false;
		for (int j = BLOCKSIZE; j < image->width - BLOCKSIZE; j++) {
			if (flag1 && isWhite(i, j - BLOCKSIZE, 0, blackAndWhiteImage) && !isWhite(i, j, 0, blackAndWhiteImage)
					&& !isWhite(i, j + BLOCKSIZE, 0, blackAndWhiteImage)) {
				flag1 = false;
				write = true;
				vecPts[saveLocation].push_back(cvPoint(j, i));
			}

			if (!flag1 && !isWhite(i, j - BLOCKSIZE, 0, blackAndWhiteImage) && isWhite(i, j, 0, blackAndWhiteImage)
					&& isWhite(i, j + BLOCKSIZE, 0, blackAndWhiteImage)) {
				flag1 = true;
				write = true;
				vecPts[saveLocation].push_back(cvPoint(j, i));
			}
		}
		if (write) saveLocation++;
	}

	// To get shoulder and largest width and coordinates
	CvPoint largest[4];

	// Assume first point is largest
	largest[0] = largest[2] = vecPts[1].at(0);
	largest[1] = largest[3] = vecPts[1].at(vecPts[1].size() - 1);

	int secondPt;

	for (int i = 10; i < saveLocation; i++) {
		int firstPt = abs(vecPts[i].at(vecPts[i].size()-1).x - vecPts[i].at(0).x);
		secondPt = abs(largest[1].x - largest[0].x);
		if (vecPts[i].size() == 2 && firstPt > secondPt) {
			largest[0] = vecPts[i].at(0);
			largest[1] = vecPts[i].at(vecPts[i].size() - 1);
		}

		if (firstPt >= secondPt) {
			largest[2] = vecPts[i - 10].at(0);
			largest[3] = vecPts[i - 10].at(vecPts[i - 10].size() - 1);
		}
	}

	critPts[SHOULDER] = largest[0];
	critPts[HAND] = largest[2];
	
	// Getting Armpit points

	for (int i = 0; i < saveLocation; i++) {
		if (vecPts[i].size() != 0 && flag == TIP) {
			flag = ARMPIT;
			critPts[TIP] = vecPts[i].at(0);
		}
		
		if (flag == ARMPIT && vecPts[i].size() == 6) {
			flag = BUFFER;
			critPts[ARMPIT] = vecPts[i].at(2);
		}
	}

	// Now Pelvis
	
	for (int i = saveLocation - 2; i >= 0; i--) {
		if (vecPts[i].size() == 4 && flag == BUFFER) {
			flag = PELVIS;
		}

		if (flag == PELVIS) {
			flag = -1;
			critPts[PELVIS] = vecPts[i].at(1);
			break;
		}
	}

	critPts[NECK] = cvPoint((critPts[TIP].x - (faceWidth / 2)), (critPts[TIP].y + faceHeight));
	critPts[WAIST] = cvPoint((critPts[PELVIS].x - (faceWidth * 2 / 3)), (critPts[PELVIS].y - faceHeight));

	int height;

	if (vecPts[saveLocation-1].at(0).y == imageHeight - 1) { 
		// if can't detect the legs, will give a rough estimate of the height
		// important as it is need for all the computation 
		height = faceHeight * HEIGHT_MULTIPLIER;
		isReal = 0;
	} else {
		height = vecPts[saveLocation-1].at(0).y - vecPts[1].at(0).y;
		isReal = 1;
	}

	// Error detection

	int midpoint = critPts[TIP].x;
	bool error = false;

	int testWidth = (critPts[SHOULDER].x - midpoint) * 2;
	testWidth = (testWidth > 0) ? testWidth : testWidth * -1;
	int testWidth2 = (critPts[HAND].x - midpoint) * 2;
	testWidth2 = (testWidth2 > 0) ? testWidth2 : testWidth2 * -1;

	if (testWidth > image->width / 2 || testWidth > faceWidth * 3.5 || testWidth <= faceWidth * 2) {
		error = true;
	} else if (testWidth2 > image->width / 2 || critPts[HAND].y > image->height || critPts[HAND].y <= critPts[SHOULDER].y * 2) {
		error = true;
	}

	// Calibration

	double calib = (double)MODEL / (double)height;

	int shoulderW = abs(critPts[TIP].x - critPts[SHOULDER].x) * calib;
	int waist = abs(critPts[TIP].x - critPts[WAIST].x) * calib;	
	
	double actualHeight = height * cmPerPixel;
	double actualShoulderWidth = shoulderW * 2 * cmPerPixel;
	double actualWaistWidth = waist * 2 * cmPerPixel;

	
	// Converting critical points to relative points

	relPts[NECK] = cvPoint((relPts[TIP].x - (critPts[TIP].x - critPts[NECK].x) * calib), 
					relPts[TIP].y + (critPts[NECK].y - critPts[TIP].y) * calib);
	relPts[SHOULDER] = cvPoint((relPts[NECK].x - (critPts[TIP].x - critPts[NECK].x) * calib), 
					relPts[NECK].y + (critPts[NECK].y - critPts[TIP].y) * calib * 0.5);
	relPts[SHOULDER].x -= faceWidth / 4;
	relPts[HAND] = cvPoint((relPts[SHOULDER].x - (critPts[SHOULDER].x - critPts[HAND].x) * calib), 
					relPts[SHOULDER].y + (critPts[HAND].y - critPts[SHOULDER].y) * calib);
	relPts[ARMPIT] = cvPoint((relPts[HAND].x + (critPts[ARMPIT].x - critPts[HAND].x) * calib), 
					relPts[HAND].y - (critPts[HAND].y - critPts[ARMPIT].y) * calib * 0.8);
	relPts[WAIST] = cvPoint((relPts[ARMPIT].x + (critPts[WAIST].x - critPts[ARMPIT].x) * calib), 
					relPts[HAND].y + (critPts[WAIST].y - critPts[ARMPIT].y) * calib * 0.5);
	relPts[PELVIS] = cvPoint(225, relPts[WAIST].y);
	
	relPts[NECK].x += faceWidth/8;
	relPts[TIP] = cvPoint(225, 0);
	relPts[HAND].y += faceHeight;
	relPts[ARMPIT].y += 10;

	// Output to file username.txt
	if (error) {
		for (int i = 1; i < 8; i++) {
			dataFile << "&point" << i << "=" << standpts[i].x << "," << standpts[i].y << endl;
		}
	} else {
		for (int i = 1; i < 8; i++) {
			dataFile << "&point" << i << "=" << relPts[i].x << "," << relPts[i].y << endl;
		}
	}

	dataFile.close();

	// Return a vector of measurements which are estimates measurements of the users

	vector<int> measurement;
	measurement.push_back(isReal);
	measurement.push_back(actualHeight);
	measurement.push_back(actualShoulderWidth);
	measurement.push_back(actualWaistWidth);

	return measurement;
}

int main(int argc, char *argv[]) {
	IplImage *imgDiff1, *imgDiff2, *blackAndWhiteImage, *endImage;
	CvSize endImageSize;
	
	// Userid. Will use this throughout the programme to identify user and retrieve images.
	char* userid = argc > 1 ? argv[1] : "1";
	char pathname_image[30] = "user/images/";
	char pathname_datapoint[30] = "user/datapoint/";

	strcat(pathname_image, userid);
	strcat(pathname_datapoint, userid);

	char filename1[80];
	char filename2[80];
	char filename3[80];
	char filename4[80];
	char filename6[80];
	char filename5[80]; // data point file
	
	strcpy(filename1,pathname_image); // background image
	strcpy(filename2,pathname_image); // background image + user
	strcpy(filename3,pathname_image); // background differenced image
	strcpy(filename4,pathname_image); // user face image
	strcpy(filename6,pathname_image);
	strcat(filename1, "1.jpg");
	strcat(filename2, "2.jpg");
	strcat(filename3, "3.jpg");
	strcat(filename4, "4.jpg");
	strcat(filename6, "6.jpg");

	strcpy(filename5,pathname_datapoint);
	strcat(filename5, ".txt");

	//printf("Filename 1: %s, Filename 2: %s\n\n", filename5, filename2);

	//printf("Processing Image...\n");
	
	imgDiff1 = cvLoadImage(filename1, 1);
	imgDiff2 = cvLoadImage(filename2, 1);
	
	// Check if Images exists..
	
	if (!imgDiff1 || !imgDiff2) {
		printf("Error: Cannot Load Image(s).");
		exit(0);
	}

	//printf("Background Differencing\n");

	endImageSize = cvSize(imgDiff1->height, imgDiff1->width);
	endImage = cvCreateImage(endImageSize, imgDiff1->depth, imgDiff1->nChannels);
	endImage = ImgPro::absDiff(imgDiff1, imgDiff2);
	
	cvSaveImage(filename3, endImage);

	//printf("Face Detection\n");

	cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name_face, 0, 0, 0 );

	if( !cascade )
    {
        fprintf( stderr, "Error: Could not load classifier cascade\n" );
        fprintf( stderr,
        "Usage: facedetect --cascade=\"<cascade_path>\" [filename|camera_index]\n" );
        return -1;
    }
    storage = cvCreateMemStorage(0);

	ImgPro::detectFace(endImage, filename4, 10, FACE);

	//printf("Pattern Recognition\n");

	cascade = (CvHaarClassifierCascade*)cvLoad( cascade_name_pattern, 0, 0, 0 );

	if( !cascade )
    {
        fprintf( stderr, "ERROR: Could not load classifier cascade\n" );
        fprintf( stderr,
        "Usage: patternDetect --cascade=\"<cascade_path>\" [filename|camera_index]\n" );
        return -1;
    }
    storage = cvCreateMemStorage(0);
	
	CvPoint cardLoc = ImgPro::detectFace(endImage, filename6, 0, PATTERN);

	//printf("Calculating Measurements\n");

	CvSize actualSize = ImgPro::getActualSize((cvLoadImage(filename6)));

	//printf("Data Point Plotting\n");

	vector<int> measurement = ImgPro::dataPoint(endImage, actualSize, filename5, filename4, cardLoc);
	
	if (measurement.at(0) == 1) {
		double height = measurement.at(1) / 1000.0;
		double shoulder = measurement.at(2) / 1000.0;
		double waist = measurement.at(3) / 1000.0;
	} else {
	}

	printf("Upload Successfully.\n");
}