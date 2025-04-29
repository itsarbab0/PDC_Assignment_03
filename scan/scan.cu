#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

// Kernel for upsweep phase
__global__ void upsweep_kernel(int* result, int two_d, int two_dplus1, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    i = i * two_dplus1;
    
    if (i + two_dplus1 - 1 < N) {
        result[i + two_dplus1 - 1] += result[i + two_d - 1];
    }
}

// Kernel for downsweep phase
__global__ void downsweep_kernel(int* result, int two_d, int two_dplus1, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    i = i * two_dplus1;
    
    if (i + two_dplus1 - 1 < N) {
        int t = result[i + two_d - 1];
        result[i + two_d - 1] = result[i + two_dplus1 - 1];
        result[i + two_dplus1 - 1] += t;
    }
}

// Kernel to set the last element to 0 (identity element for sum)
__global__ void set_last_element_kernel(int* result, int N) {
    result[N-1] = 0;
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result

///////==========================================================================================================

//////   Ab is part me hamny scan.cu me hamain prefix sum lena thaa 
//////  is k liye ye function aur is k neechy wala repeate ka funtion likhna thaa
void exclusive_scan(int* input, int N, int* result)
{
    // We're already given the input copied to result, so we can perform an in-place scan

    // Round up to next power of 2
    int rounded_N = nextPow2(N);
    
    // upsweep phase
    for (int two_d = 1; two_d <= rounded_N/2; two_d *= 2) {
        int two_dplus1 = 2 * two_d;
        int num_blocks = (rounded_N + two_dplus1 - 1) / two_dplus1;
        int threads_per_block = THREADS_PER_BLOCK;
        
        // Ensure we don't have empty blocks
        if (num_blocks > 0) {
            upsweep_kernel<<<num_blocks, threads_per_block>>>(result, two_d, two_dplus1, rounded_N);
            cudaDeviceSynchronize();
        }
    }
    
    // Set the last element to 0 (identity element for sum)
    set_last_element_kernel<<<1, 1>>>(result, rounded_N);
    cudaDeviceSynchronize();
    
    // downsweep phase
    for (int two_d = rounded_N/2; two_d >= 1; two_d /= 2) {
        int two_dplus1 = 2 * two_d;
        int num_blocks = (rounded_N + two_dplus1 - 1) / two_dplus1;
        int threads_per_block = THREADS_PER_BLOCK;
        
        // Ensure we don't have empty blocks
        if (num_blocks > 0) {
            downsweep_kernel<<<num_blocks, threads_per_block>>>(result, two_d, two_dplus1, rounded_N);
            cudaDeviceSynchronize();
        }
    }
}

// Kernel for marking repeats: sets 1 where input[i] == input[i+1], 0 otherwise
__global__ void mark_repeats_kernel(int* input, int* output, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < length - 1) { // Ensure we don't go beyond array bounds
        output[idx] = (input[idx] == input[idx + 1]) ? 1 : 0;
    } else if (idx == length - 1) {
        output[idx] = 0; // Last element can't have a repeat
    }
}

// Kernel to write the output indices where repeats were found
__global__ void write_repeats_indices_kernel(int* flags, int* indices, int length) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < length - 1 && flags[idx] == 1) {
        indices[flags[idx + length - 1]] = idx;
    }
}

// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {
    int rounded_length = nextPow2(length);
    
    // Temporary storage for flags and scanned flags
    int* device_flags;
    cudaMalloc(&device_flags, rounded_length * sizeof(int));
    
    // Set flags where input[i] == input[i+1]
    int blocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    mark_repeats_kernel<<<blocks, THREADS_PER_BLOCK>>>(device_input, device_flags, length);
    cudaDeviceSynchronize();
    
    // Perform exclusive scan on the flags
    exclusive_scan(device_flags, rounded_length, device_flags);
    
    // Copy the total count (last element + last flag)
    int total_repeats = 0;
    int last_flag = 0;
    
    // Get the last flag value (1 or 0)
    if (length > 0) {
        cudaMemcpy(&last_flag, &device_flags[length-1], sizeof(int), cudaMemcpyDeviceToHost);
    }
    
    // Get the total from the last element of the scan
    if (length > 0) {
        int temp = 0;
        cudaMemcpy(&temp, &device_flags[length], sizeof(int), cudaMemcpyDeviceToHost);
        total_repeats = temp;
    }
    
    // Adjust for the last element if it's a repeat
    if (length > 0) {
        total_repeats = total_repeats + last_flag;
    }
    
    // If no repeats found, cleanup and return
    if (total_repeats == 0) {
        cudaFree(device_flags);
        return 0;
    }
    
    // Write the output indices
    write_repeats_indices_kernel<<<blocks, THREADS_PER_BLOCK>>>(device_flags, device_output, length);
    cudaDeviceSynchronize();
    
    cudaFree(device_flags);
    
    return total_repeats;
}
///////  Yahan tak ka part hamny karna tha
///////==========================================================================================================

//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;
    int N = end - inarray;  

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);
    
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
    // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, N, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
       
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
   
    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);

    thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration; 
}


//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {

    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime; 
    return duration;
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}
