
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <stdio.h>

#include<ctime>


using namespace cv;
using namespace std;

void cpuThresholding(Mat& output, const Mat& input, unsigned char th)
{
	// TODO: Add your CPU side serial code to perform thresholding here
	
	int rows = input.rows;
	int cols = input.cols;
	for (int i=0;i<rows;i++)
		for (int j = 0; j < cols; j++)
		{
			int index = j + i*cols;
			if (input.data[index] < th)
				output.data[index] = 0;
			else
				output.data[index] = 255;
		
		}
	
}




//CUDA function prototype. It takes output image and input image and a threshold value
cudaError_t  performWithCuda(Mat&, const Mat&, unsigned char th);

// CUDA GPU Kernel 
__global__ void gpukernel(unsigned char *dst, const unsigned char * src, unsigned char th)
{
	// TODO: 
	//  1- calculate the index of the pointers based on pixel location for each thread
	//	2- perform the thresholding
	int index = threadIdx.x + blockIdx.x*blockDim.x;
	if (src[index] < th)

		dst[index] = 0;
	else
		dst[index] = 255;

}

int main()
{
	unsigned char threshold = 128;		// This is a threshold value, you can change this value
	cudaError_t cudaStatus;			// This is the cudaError code that your functions may return to troubleshoot



									//	TODO: 
									//	1- Read the input gray-scale image with imread
	Mat inputImage = imread("C:\\Users\\sw5\\Desktop\\course books\\6398\\Home works\\Project 1\\Hydrangeas_gray.jpg",0);
	//1 - 2 - 1 - create an image for the CPU output, and one for the GPU output
	Mat cpuThresholdImage(inputImage.rows, inputImage.cols, CV_8UC1);
	Mat gpuThresholdImage(inputImage.rows, inputImage.cols, CV_8UC1);
									
								
	//		1-1- if image has no data show an error message
	if (!inputImage.data)
		printf("Image didn't load properly!\n");
	else
	{
		//		    1-2- if iamge has data
		//			1-2-2- call your CPU  side code to threshold the image (pass the input image and the cpu output image and the threshold)
		cpuThresholding(cpuThresholdImage, inputImage, threshold);
		
		// 1 - 2 - 3 - call the performWithCuda function to create gpu pointers, copy data from host to device, invoke kernel
			//						and copy results back to host (refer to the above function prototype on line 23 for reference.)
		cudaStatus = performWithCuda(gpuThresholdImage, inputImage, threshold);
		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "Cuda failed");
			return 1;
		}
		//1 - 2 - 4 - Use imshow to show the input image, the CPU output and the GPU output.Note: CPU and GPU outputs should look alike.
		imshow("Input_Image", inputImage);
		imshow("CPU_OUTPUT", cpuThresholdImage);
		imshow("GPU_OUTPUT", gpuThresholdImage);
	}

		
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	//		DO NOT CHANGE THE FOLLOWING!
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	//		1-3- use cvWaitKey(0); to pause.
	cvWaitKey(0);


	return 0;
}

// Helper function for using CUDA to perform image thresholding in parallel. Takes as input the thresholded image (bwImage), the input image (input), and the threshold value.
cudaError_t performWithCuda(Mat & outputImg, const Mat & inputImg, unsigned char threshold)
{
	
	// Allocate GPU buffers for the buffers (one input, one output)   
	unsigned char *dev_ptrout, *dev_ptrin;	// these are the gpu side ouput and input pointers

	cudaError_t cudaStatus;



	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	
	
	// TODO: add your code here to allocate the input pointer on the device. Note the size of the pointer in cudaMalloc
	cudaStatus = cudaMalloc((void**)& dev_ptrin, sizeof(unsigned char)*inputImg.rows*inputImg.cols);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Cuda failed");
		goto Error;
	}

	// TODO: add your code here to allocate the outpu pointer on the device. Note the size of the pointer in cudaMalloc
	cudaStatus = cudaMalloc((void**)& dev_ptrout, sizeof(unsigned char)*outputImg.rows*outputImg.cols);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Cuda failed");
		goto Error;
	}
	// Copy input data from host memory to GPU buffers.
	// TODO: Add your code here. Use cudaMemcpy
	cudaStatus = cudaMemcpy(dev_ptrin, inputImg.data, sizeof(unsigned char)*inputImg.rows*inputImg.cols, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "Cuda failed");
		goto Error;
	}

	// TODO: Launch a kernel on the GPU with one thread for each element. use <<< grid_size (or number of blocks), block_size(or number of threads) >>>
	int grid_size=inputImg.rows;
	int block_size=inputImg.cols;
	// lauch a kernel on the GPU with one thread for each element.
	gpukernel << <grid_size, block_size >> > (dev_ptrout, dev_ptrin,threshold);
	
	// Check for any errors launching the kernel
	cudaStatus= cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// TODO: Copy output data from GPU buffer to host memory. use cudaMemcpy
	cudaStatus = cudaMemcpy(outputImg.data, dev_ptrout, sizeof(unsigned char)*outputImg.rows*outputImg.cols, cudaMemcpyDeviceToHost);

		if (cudaStatus != cudaSuccess)
		{
			fprintf(stderr, "Cuda failed");
			goto Error;
		}
Error:
	cudaFree(dev_ptrin);
	cudaFree(dev_ptrout);

	return cudaStatus;
}