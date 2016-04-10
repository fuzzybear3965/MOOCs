#include <wb.h>

#define TILE_WIDTH 32
#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int numARows, 
									 int numAColumns, int numBRows, int numBColumns,
									 int numCRows,int numCColumns){ 
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to use shared memory for this MP
  int tx = threadIdx.x ; int ty = threadIdx.y;
  int bx = blockIdx.x  ; int by = blockIdx.y;
  __shared__ float tempA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float tempB[TILE_WIDTH][TILE_WIDTH];
   
  int Row = ty+by*TILE_WIDTH;
  int Col = tx+bx*TILE_WIDTH;
	
  float Cvalue = 0;
	for (int i = 0; i < ((numAColumns-1)/TILE_WIDTH)+1 ; ++i){
		if(Row < numARows && i*TILE_WIDTH+tx < numAColumns){
    		tempA[ty][tx] = A[(Row*numAColumns)+(i*TILE_WIDTH)+tx];
			} else{tempA[ty][tx] = 0.0;}
		
		if(i*TILE_WIDTH+ty < numBRows  && Col < numBColumns){
    		tempB[ty][tx] = B[(i*TILE_WIDTH+ty)*numBColumns+Col];
		   	}else {tempB[ty][tx] = 0.0;}
		
  		__syncthreads();
		
  		for (int k = 0; k < TILE_WIDTH; ++k) {
    		Cvalue += tempA[ty][k] * tempB[k][tx];
  		}
		
	  	__syncthreads();
		
		if(Row < numCRows && Col < numCColumns){ C[Row*numCColumns + Col] = Cvalue;}
	}
}


int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set this)
  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA =
      ( float * )wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
  hostB =
      ( float * )wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows = numARows;
  numCColumns = numBColumns;
  //@@ Allocate the hostC matrix
  //hostC = (float*) malloc(sizeof(float)*numCColumns*numCRows);
  hostC = new float[numCRows*numCColumns];

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  wbCheck(cudaMalloc((void**)&deviceA,sizeof(float)*numARows*numAColumns));
  wbCheck(cudaMalloc((void**)&deviceB,sizeof(float)*numBRows*numBColumns));
  wbCheck(cudaMalloc((void**)&deviceC,sizeof(float)*numCRows*numCColumns));
 
  wbTime_stop(GPU, "Allocating GPU memory.");
  
  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
    cudaMemcpy(deviceA,hostA,sizeof(float)*numARows*numAColumns,cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB,hostB,sizeof(float)*numBRows*numBColumns,cudaMemcpyHostToDevice);
    cudaMemcpy(deviceC,hostC,sizeof(float)*numCRows*numCColumns,cudaMemcpyHostToDevice);

  wbTime_stop(GPU, "Copying input memory to the GPU.");
   printf("C is : %d x %d\n",numCRows, numCColumns);
  //@@ Initialize the grid and block dimensions here
    dim3 griddim(ceil(numCColumns/(float)TILE_WIDTH),ceil(numCRows/(float)TILE_WIDTH),1);
//	int numhoriztiles = ceil(numCColumns/(float)TILE_WIDTH);
//  int numverttiles = ceil(numCRows/(float)TILE_WIDTH);
//	printf("griddim: %f x %f x 1\n",numhoriztiles,numverttiles);
    dim3 blockdim(TILE_WIDTH,TILE_WIDTH,1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
    matrixMultiplyShared <<< griddim, blockdim >>> (deviceA,deviceB,deviceC,numARows,numAColumns
													,numBRows,numBColumns,numCRows,numCColumns);
  
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  cudaMemcpy(hostC,deviceC,sizeof(float)*numCRows*numCColumns,cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
//  	if ( deviceC != NULL )
		cudaFree(deviceC); 
//	if ( deviceB != NULL )
		cudaFree(deviceB);
//	if (deviceA != NULL)
		cudaFree(deviceA);
	
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}