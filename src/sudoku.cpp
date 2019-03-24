#include <iostream>
#include <fstream>
#include <math.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2\imgproc\types_c.h> 
#include <stdlib.h>
#include <stdio.h>
#include <direct.h>
#include <windows.h>
#include "sudoku.h"

using namespace cv;
using namespace std;

struct box_st
{
	Point real_pos;
	Point table_pos;
	char value;
	Mat pic;
	bool isnull;
	static Size box_size;
};

Size box_st::box_size;

static void cutNums(struct box_st boxes[])
{
	Mat pic;
	int rows = 0, cols = 0;
	int max_size = 0, add_pixel1 = 0, add_pixel2 = 0;

	for (int i = 0; i < 81; i++)
	{
		if (boxes[i].isnull == false)
		{
			pic = boxes[i].pic.clone();
			rows = boxes[i].pic.rows;
			cols = boxes[i].pic.cols;
			add_pixel1 = abs(rows - cols) / 2;
			add_pixel2 = abs(rows - cols) - add_pixel1;
			if (rows > cols)
			{
				copyMakeBorder(pic, pic, 0, 0, add_pixel1, add_pixel2, BORDER_CONSTANT, Scalar(255));
			}
			else if (cols > rows)
			{
				copyMakeBorder(pic, pic, add_pixel1, add_pixel2, 0, 0, BORDER_CONSTANT, Scalar(255));
			}
			imwrite(format(".\\nums_data\\%d.jpg", i), pic);
		}
	}
}

static bool findSudokuNums(Mat& src, struct box_st ans[], bool show = false, bool draw = false)
{
	Mat bin(src.rows, src.cols, CV_8UC1);

	int kernel_size = src.cols > src.rows ? src.cols / 3 : src.rows / 3;
	kernel_size = (kernel_size % 2) ? kernel_size : (kernel_size + 1);

	adaptiveThreshold(src, bin, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, kernel_size, 10);

	vector<vector<Point>>contours;
	vector<Vec4i>hierarchy;
	findContours(bin, contours, hierarchy, RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	Moments mu;
	int j = 0;
	vector<Rect> boundRect;
	for (int i = 0; i < hierarchy.size(); i++)
	{
		if (hierarchy[i][3] == hierarchy[0][2])
		{
			if (j == 81)
			{
				break;
				j = 88;
			}
			if (hierarchy[i][2] != -1)
			{
				boundRect.push_back(boundingRect(Mat(contours[hierarchy[i][2]])));
				ans[j].pic = src(boundRect[boundRect.size() - 1]).clone();
				ans[j].isnull = false;
			}
			else
			{
				ans[j].isnull = true;
			}

			mu = moments(contours[i]);
			ans[j].real_pos = Point(mu.m10 / mu.m00, mu.m01 / mu.m00);
			j++;
		}
	}

	if (show == true)
	{
		Mat contours_pic(bin.size(), CV_8UC3);
		drawContours(contours_pic, contours, -1, Scalar(255, 0, 0), 1, 8, hierarchy);
		imshow("bin", bin);
		imshow("contours", contours_pic);
		waitKey(1);
	}

	if (draw == true)
	{

	}

	if (j == 81)
	{
		return true;
	}
	else
	{
		return false;
	}

}

static bool numeralRecognition(struct box_st boxes[])
{
	Mat pic;
	int rows = 0, cols = 0;
	int max_size = 0, add_pixel1 = 0, add_pixel2 = 0;

	char path_buf[256];
	string path;
	_getcwd(path_buf, sizeof(path_buf));
	path = format("del %s\\test_buf\\*.jpg", path_buf);
	system(path.c_str());

	for (int i = 0; i < 81; i++)
	{
		if (boxes[i].isnull == false)
		{
			pic = boxes[i].pic.clone();
			rows = boxes[i].pic.rows;
			cols = boxes[i].pic.cols;
			add_pixel1 = abs(rows - cols) / 2;
			add_pixel2 = abs(rows - cols) - add_pixel1;
			if (rows > cols)
			{
				copyMakeBorder(pic, pic, 0, 0, add_pixel1, add_pixel2, BORDER_CONSTANT, Scalar(255));
			}
			else if (cols > rows)
			{
				copyMakeBorder(pic, pic, add_pixel1, add_pixel2, 0, 0, BORDER_CONSTANT, Scalar(255));
			}
			resize(pic, pic, Size(19, 19));
			imwrite(format(".\\test_buf\\%d.jpg", i), pic);
		}
		else
		{
			boxes[i].value = 0;
		}
	}

	path = format("python %s\\number_load_model.py", path_buf);
	system(path.c_str());

	fstream file;
	char txt_buf[256];
	string str;
	int index;
	path = format("%s\\answer.csv", path_buf);
	file.open(path.c_str());
	file.getline(txt_buf, 256);

	while (!file.eof())
	{
		file.getline(txt_buf, 256);
		str = txt_buf;
		index = str.find(".jpg");
		if (index < 100 && index > 0)
		{
			txt_buf[index] = '\0';
			boxes[atoi(txt_buf)].value = atoi(&txt_buf[index + 5]);
		}
	}

	return false;
}

static void coverntToTable(struct box_st boxes[], Mat& table)
{
	Point max_pos = boxes[0].real_pos;
	Point min_pos = max_pos;
	double max = norm(boxes[0].real_pos);
	double min = max;
	double pos_norm;

	for (int i = 0; i < 81; i++)
	{
		pos_norm = norm(boxes[i].real_pos);
		if (pos_norm > max)
		{
			max = pos_norm;
			max_pos = boxes[i].real_pos;
		}

		if (pos_norm < min)
		{
			min = pos_norm;
			min_pos = boxes[i].real_pos;
		}
	}

	Size box_size((max_pos.x - min_pos.x) / 8, (max_pos.y - min_pos.y) / 8);
	boxes[0].box_size = box_size;

	int r, c;
	for (int i = 0; i < 81; i++)
	{
		r = round((boxes[i].real_pos.y - min_pos.y) / (float)box_size.height);
		c = round((boxes[i].real_pos.x - min_pos.x) / (float)box_size.width);
		table.at<unsigned char>(r, c) = boxes[i].value;
		boxes[i].table_pos = Size(r, c);
	}
}

void showSudoku(Mat& src, Mat& ans, box_st boxes[], String nums_pic_path, bool show = false)
{
	Mat paste_roi;
	vector<Mat> nums_pic;
	int max_size = 0;
	Point delta_pixels;
	Point roi_pos_1;
	Point roi_pos_2;
	Size pic_size(0, 0);

	for (int i = 0; i < 81; i++)
	{
		if (boxes[i].isnull == false)
		{
			pic_size = boxes[i].pic.size();
			max_size = pic_size.width > pic_size.height ? pic_size.width : pic_size.height;
			break;
		}
	}

	if (max_size == 0)
	{
		pic_size = boxes[0].box_size;
		max_size = pic_size.width > pic_size.height ? (pic_size.height * 3) / 4 : (pic_size.width * 3) / 4;
	}

	delta_pixels = Point(max_size / 2, max_size - max_size / 2);

	for (int i = 0; i < 9; i++)
	{
		nums_pic.push_back(imread(nums_pic_path + format("%d.jpg", i + 1), IMREAD_GRAYSCALE));
		resize(nums_pic[i], nums_pic[i], Size(max_size, max_size));
	}

	for (int i = 0; i < 81; i++)
	{
		if (boxes[i].isnull == true)
		{
			boxes[i].value = ans.at<unsigned char>(boxes[i].table_pos.x, boxes[i].table_pos.y);
			if (boxes[i].value < 1)
			{
				break;
			}
			roi_pos_1 = boxes[i].real_pos - delta_pixels;
			roi_pos_2 = roi_pos_1 + Point(max_size, max_size);
			paste_roi = src(Rect(roi_pos_1, roi_pos_2));
			resize(paste_roi, paste_roi, Size(max_size, max_size));
			nums_pic[boxes[i].value - 1].copyTo(paste_roi);
		}
	}

	if (show == true)
	{
		cout << ans << endl;
		imshow("src", src);
		waitKey(1);
	}
}

bool __solve_sudoku(char sudoku[9][9], int position, bool row_rule[9][9], bool col_rule[9][9], bool sec_rule[9][9])
{
	int index = position;
	for (; index < 81 && 0 != sudoku[index / 9][index % 9]; index++);

	if (index < 81)
	{
		int r = index / 9;
		int c = index % 9;

		for (int i = 0; i < 9; i++)
		{
			if (row_rule[r][i] || col_rule[c][i] || sec_rule[(r / 3) * 3 + c / 3][i])
				continue;

			sudoku[r][c] = i + 1;
			row_rule[r][i] = true;
			col_rule[c][i] = true;
			sec_rule[(r / 3) * 3 + c / 3][sudoku[r][c] - 1] = true;

			if (__solve_sudoku(sudoku, index + 1, row_rule, col_rule, sec_rule))
				return true;

			sudoku[r][c] = 0;
			row_rule[r][i] = false;
			col_rule[c][i] = false;
			sec_rule[(r / 3) * 3 + c / 3][i] = false;
		}
		return false;
	}
	return true;
}

bool solveSudoku(Mat& src, Mat& ans, bool show, bool draw, String nums_pic_path)
{
	box_st boxes[81];
	bool row_rule[9][9];
	bool col_rule[9][9];
	bool sec_rule[9][9];
	char sudoku[9][9];

	findSudokuNums(src, boxes, show);
	//cutNums(boxes);
	numeralRecognition(boxes);
	coverntToTable(boxes, ans);

	//ans.at<unsigned char>(0, 0) = 5;
	//ans.at<unsigned char>(0, 1) = 3;
	//ans.at<unsigned char>(0, 4) = 7;
	//ans.at<unsigned char>(1, 0) = 6;
	//ans.at<unsigned char>(1, 3) = 1;
	//ans.at<unsigned char>(1, 4) = 9;
	//ans.at<unsigned char>(1, 5) = 5;
	//ans.at<unsigned char>(2, 1) = 9;
	//ans.at<unsigned char>(2, 2) = 8;
	//ans.at<unsigned char>(2, 7) = 6;
	//ans.at<unsigned char>(3, 0) = 8;
	//ans.at<unsigned char>(3, 4) = 6;
	//ans.at<unsigned char>(3, 8) = 3;
	//ans.at<unsigned char>(4, 0) = 4;
	//ans.at<unsigned char>(4, 3) = 8;
	//ans.at<unsigned char>(4, 5) = 3;
	//ans.at<unsigned char>(4, 8) = 1;
	//ans.at<unsigned char>(5, 0) = 7;
	//ans.at<unsigned char>(5, 4) = 2;
	//ans.at<unsigned char>(5, 8) = 6;
	//ans.at<unsigned char>(6, 1) = 6;
	//ans.at<unsigned char>(6, 6) = 2;
	//ans.at<unsigned char>(6, 7) = 8;
	//ans.at<unsigned char>(7, 3) = 4;
	//ans.at<unsigned char>(7, 4) = 1;
	//ans.at<unsigned char>(7, 5) = 9;
	//ans.at<unsigned char>(7, 8) = 5;
	//ans.at<unsigned char>(8, 4) = 8;
	//ans.at<unsigned char>(8, 7) = 7;
	//ans.at<unsigned char>(8, 8) = 9;

	for (int r = 0; r < 9; r++)
	{
		for (int c = 0; c < 9; c++)
		{
			row_rule[r][c] = false;
			col_rule[r][c] = false;
			sec_rule[r][c] = false;
		}
	}

	for (int r = 0; r < 9; r++)
	{
		for (int c = 0; c < 9; c++)
		{
			sudoku[r][c] = ans.at<unsigned char>(r, c);
			if (sudoku[r][c] != 0)
			{
				row_rule[r][sudoku[r][c] - 1] = true;
				col_rule[c][sudoku[r][c] - 1] = true;
				sec_rule[(r / 3) * 3 + c / 3][sudoku[r][c] - 1] = true;
			}
		}
	}

	if (true == __solve_sudoku(sudoku, 0, row_rule, col_rule, sec_rule))
	{
		for (int r = 0; r < 9; r++)
		{
			for (int c = 0; c < 9; c++)
			{
				ans.at<unsigned char>(r, c) = sudoku[r][c];
			}
		}
	}

	if (draw == true)
	{
		showSudoku(src, ans, boxes, nums_pic_path, show);
	}

	return false;
}







