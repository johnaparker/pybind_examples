/*
Compile with:

g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` fft.cpp -o fft`python3-config --extension-suffix` -I./pybind11/include -I/usr/include/python3.7m -I/opt/cuda/include -L/opt/cuda/lib64 -lcudart -lcufft

*/

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdlib.h>
#include <pybind11/pybind11.h>

using namespace pybind11::literals;

float getRand(void)
{
    return (float)(rand() % 16);
}

void gpu_fft(const int dataH, const int dataW, const int iterations) {
    float *h_Data;
    float *d_Data;
    cuComplex *d_DataSpectrum;

    h_Data = (float *)malloc(dataH*dataW * sizeof(float));

    cudaMalloc((void **)&d_Data, dataH*dataW * sizeof(float));
    cudaMalloc((void **)&d_DataSpectrum,   dataH * (dataW / 2 + 1) * sizeof(cuComplex));

    for (int i = 0; i < dataH * dataW; i++)
    {
        h_Data[i] = getRand();
    }
    cudaMemcpy(d_Data, h_Data,   dataH*dataW * sizeof(float), cudaMemcpyHostToDevice);

    cufftHandle fftPlanFwd;
    cufftPlan2d(&fftPlanFwd, dataH, dataW, CUFFT_R2C);

    for (int i = 0; i < iterations; i++) {
        cufftExecR2C(fftPlanFwd, (cufftReal *)d_Data, (cufftComplex *)d_DataSpectrum);
        cudaDeviceSynchronize();
    }
    //modulateAndNormalize(d_DataSpectrum, d_KernelSpectrum, fftH, fftW, 1);

    free(h_Data);

    cufftDestroy(fftPlanFwd);
    cudaFree(d_Data);
    cudaFree(d_DataSpectrum);
}

PYBIND11_MODULE(fft, m) {
    m.doc() = "FFT on the GPU";
    m.def("gpu_fft", gpu_fft, 
            "dataH"_a, "dataW"_a, "iterations"_a, R"pbdoc(
        fft on GPU
    )pbdoc");
}
