#pragma once
#ifndef SUDOKU
#define SUDOKU


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

bool solveSudoku(cv::Mat& src, cv::Mat& ans, bool show = false, bool draw = false, cv::String nums_pic_path = "./nums_pic/");

#endif // SUDOKU