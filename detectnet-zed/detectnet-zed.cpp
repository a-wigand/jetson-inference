/*
 * Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "detectNet.h"
#include "loadImage.h"

#include "cudaMappedMemory.h"
#include "cudaNormalize.h"
#include "cudaFont.h"

#include "gstCamera.h"
#include "glDisplay.h"
#include "glTexture.h"

#include <sys/time.h>
#include <stdio.h>
#include <signal.h>
#include <unistd.h>

#define DEFAULT_CAMERA 1 

bool signal_recieved = false;

void sig_handler(int signo)
{
        if( signo == SIGINT )
        {
                printf("received SIGINT\n");
                signal_recieved = true;
        }
}

uint64_t current_timestamp() {
    struct timeval te; 
    gettimeofday(&te, NULL); // get current time
    return te.tv_sec*1000LL + te.tv_usec/1000; // caculate milliseconds
}

// main entry point
int main( int argc, char** argv )
{
	printf("detectnet-zed\n  args (%i):  ", argc);
	
	for( int i=0; i < argc; i++ )
		printf("%i [%s]  ", i, argv[i]);
		
	printf("\n\n");
	
        if( signal(SIGINT, sig_handler) == SIG_ERR )
                printf("\ncan't catch SIGINT\n");
	
	const char* imgFilename = argv[1];

        /*
         * create the camera device
         */
        gstCamera* camera = gstCamera::Create(DEFAULT_CAMERA);

        if( !camera )
        {
                printf("\ndetectnet-zed:  failed to initialize video device\n");
                return 0;
        }

        printf("\ndetectnet-zed:  successfully initialized video device\n");
        printf("    width:  %u\n", camera->GetWidth());
        printf("   height:  %u\n", camera->GetHeight());
        printf("    depth:  %u (bpp)\n\n", camera->GetPixelDepth());


	// create detectNet
	detectNet* net = detectNet::Create(argc, argv);

	if( !net )
	{
		printf("detectnet-zed:   failed to initialize detectNet\n");
		return 0;
	}

	// net->EnableProfiler();
	
	// alloc memory for bounding box & confidence value output arrays
	const uint32_t maxBoxes = net->GetMaxBoundingBoxes();
	const uint32_t classes  = net->GetNumClasses();
	
	float* bbCPU    = NULL;
	float* bbCUDA   = NULL;
	float* confCPU  = NULL;
	float* confCUDA = NULL;
	
	if( !cudaAllocMapped((void**)&bbCPU, (void**)&bbCUDA, maxBoxes * sizeof(float4)) ||
	    !cudaAllocMapped((void**)&confCPU, (void**)&confCUDA, maxBoxes * classes * sizeof(float)) )
	{
		printf("detectnet-zed:  failed to alloc output memory\n");
		return 0;
	}
	
        /*
         * create openGL texture
         */
//        glTexture* texture = NULL;

//        texture = glTexture::Create(camera->GetWidth(), camera->GetHeight(), GL_RGBA32F_ARB/*GL_RGBA8*/);

//        if( !texture )
//                printf("detectnet-zed:  failed to create openGL texture\n");

        /*
         * create font
         */
        cudaFont* font = cudaFont::Create();

        /*
         * start camera
         */
        if( !camera->Open() )
        {
                printf("\ndetectnet-zed:  failed to start camera\n");
                return 0;
        }

        printf("\ndetectnet-zed:  camera started\n");

        float confidence = 0.0f;

	if( !signal_recieved )
	{
                void* imgCPU  = NULL;
                void* imgCUDA = NULL;

                // get the latest frame
                if( !camera->Capture(&imgCPU, &imgCUDA, 1000) )
                        printf("\ndetectnet-camera:  failed to capture frame\n");

                // convert from YUV to RGBA
                void* imgRGBA = NULL;

                if( !camera->ConvertRGBA(imgCUDA, &imgRGBA) )
                        printf("detectnet-camera:  failed to convert from NV12 to RGBA\n");

                // classify image with detectNet
                int numBoundingBoxes = maxBoxes;
	}

	// load image from file on disk
//	float* imgCPU    = NULL;
//	float* imgCUDA   = NULL;
//	int    imgWidth  = 0;
//	int    imgHeight = 0;
		
//	if( !loadImageRGBA(imgFilename, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
//	{
//		printf("failed to load image '%s'\n", imgFilename);
//		return 0;
//	}

	void* imgCPU  = NULL;
        void* imgCUDA = NULL;

        // get the latest frame
        if( !camera->Capture(&imgCPU, &imgCUDA, 1000) )
                printf("\ndetectnet-camera:  failed to capture frame\n");

        // convert from YUV to RGBA
        void* imgRGBA = NULL;

        if( !camera->ConvertRGBA(imgCUDA, &imgRGBA) )
                printf("detectnet-camera:  failed to convert from NV12 to RGBA\n");

	
	// classify image
	int numBoundingBoxes = maxBoxes;
	
	printf("detectnet-zed:  beginning processing network (%zu)\n", current_timestamp());

//        const bool result = net->Detect(imgCUDA, imgWidth, imgHeight, bbCPU, &numBoundingBoxes, confCPU);
	const bool result = net->Detect((float*)imgRGBA, camera->GetWidth(), camera->GetHeight(), bbCPU, &numBoundingBoxes, confCPU);

	printf("detectnet-zed:  finished processing network  (%zu)\n", current_timestamp());

	if( !result )
		printf("detectnet-zed:  failed to classify '%s'\n", imgFilename);
	else if( argc > 2 )		// if the user supplied an output filename
	{
		printf("%i bounding boxes detected\n", numBoundingBoxes);
		
		int lastClass = 0;
		int lastStart = 0;
		
		for( int n=0; n < numBoundingBoxes; n++ )
		{
			const int nc = confCPU[n*2+1];
			float* bb = bbCPU + (n * 4);
			
			printf("detected obj %i  class #%u (%s)  confidence=%f\n", n, nc, net->GetClassDesc(nc), confCPU[n*2]);
			printf("bounding box %i  (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, bb[0], bb[1], bb[2], bb[3], bb[2] - bb[0], bb[3] - bb[1]); 
			
			if( nc != lastClass || n == (numBoundingBoxes - 1) )
			{
//                                if( !net->DrawBoxes(imgCUDA, imgCUDA, imgWidth, imgHeight, bbCUDA + (lastStart * 4), (n - lastStart) + 1, lastClass) )
				if( !net->DrawBoxes((float*)imgRGBA, (float*)imgRGBA, camera->GetWidth(), camera->GetHeight(), bbCUDA + (lastStart * 4), (n - lastStart) + 1, lastClass) )
					printf("detectnet-zed:  failed to draw boxes\n");
					
				lastClass = nc;
				lastStart = n;
			}
		}
		
		CUDA(cudaThreadSynchronize());
		
		// save image to disk
		printf("detectnet-zed:  writing %ix%i image to '%s'\n", camera->GetWidth(), camera->GetHeight(), argv[2]);
		
//                if( !saveImageRGBA(argv[2], (float4*)imgCPU, imgWidth, imgHeight, 255.0f) )
		// should be imgRGBA
		if( !saveImageRGBA(argv[2], (float4*)imgCPU, camera->GetWidth(), camera->GetHeight(), 255.0f) )
			printf("detectnet-zed:  failed saving %ix%i image to '%s'\n", camera->GetWidth(), camera->GetHeight(), argv[2]);
		else	
			printf("detectnet-zed:  successfully wrote %ix%i image to '%s'\n", camera->GetWidth(), camera->GetHeight(), argv[2]);
		
	}
	//printf("detectnet-console:  '%s' -> %2.5f%% class #%i (%s)\n", imgFilename, confidence * 100.0f, img_class, "pedestrian");

        /*
         * shutdown the camera device
         */
        if( camera != NULL )
        {
                delete camera;
                camera = NULL;
        }

	printf("Debug: Make successful!");
	printf("\nshutting down...\n");
	CUDA(cudaFreeHost(imgCPU));
	delete net;
	return 0;
}
