using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace cudaFLAB
{
    class cuFLAB
    {
        public const int BLOCK_SIZE = 32;

        /*
        public static void cuMatrixVectorProd(float[] dA, float[] dx, float[] dy, long nRows, long nCols)
        {
            var tid = blockIdx.x * blockDim.x + threadIdx.x;

            float[] x_shared = __shared__.Array<float>(BLOCK_SIZE);

            float y_val = 0f;

            for (int m = 0; m < ((nCols + BLOCK_SIZE - 1) / BLOCK_SIZE); ++m)
            {
                if ((m * BLOCK_SIZE + threadIdx.x) < nCols) x_shared[threadIdx.x] = dx[threadIdx.x + m * BLOCK_SIZE];
                else x_shared[threadIdx.x] = 0f;
                DeviceFunction.SyncThreads();

                for (int e = 0; e < BLOCK_SIZE; ++e)
                {
                    // --- Column-major ordering - faster
                    y_val += dA[tid + (e + BLOCK_SIZE * m) * nRows] * x_shared[e];
                    // --- Row-major ordering - slower
                    //y_val += dA[tid * nCols + (e + blockSize * m)] * x_shared[e];
                }

                DeviceFunction.SyncThreads();
            }

            if (tid < nRows) dy[tid] = y_val;
        }

        public static void cuMatrixTensorProd(float[] A, float[] B, float[] C, long ARows, long ACols, long BRows, long BCols, long CRows, long CCols)
        {
            float CValue = 0;

            int Row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
            int Col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

            float[] As = __shared__.Array<float>(BLOCK_SIZE * BLOCK_SIZE);
            float[] Bs = __shared__.Array<float>(BLOCK_SIZE * BLOCK_SIZE);

            for (int k = 0; k<(BLOCK_SIZE + ACols - 1)/BLOCK_SIZE; k++) {
                if (k* BLOCK_SIZE + threadIdx.x<ACols && Row<ARows)
                {
                    As[threadIdx.y * BLOCK_SIZE + threadIdx.x] = A[Row * ACols + k * BLOCK_SIZE + threadIdx.x];
                }                                        
                else
                {
                    As[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0f;
                }
                         
                if (k* BLOCK_SIZE + threadIdx.y<BRows && Col<BCols)
                {
                    Bs[threadIdx.y * BLOCK_SIZE + threadIdx.x] = B[(k * BLOCK_SIZE + threadIdx.y) * BCols + Col];
                }
                else
                {
                    Bs[threadIdx.y * BLOCK_SIZE + threadIdx.x] = 0f;
                }

                DeviceFunction.SyncThreads();

                for (int n = 0; n<BLOCK_SIZE; ++n)
                {
                    CValue += As[threadIdx.y * BLOCK_SIZE + n] * Bs[n];
                }

                DeviceFunction.SyncThreads();

            }

            if (Row<CRows && Col<CCols) {
                C[((blockIdx.y * blockDim.y + threadIdx.y) * CCols) + (blockIdx.x * blockDim.x) + threadIdx.x] = CValue;
            }                    
        }

        public static void cuAdd(float[] a, float[] b, float[] c)
        {
            var start = blockIdx.x * blockDim.x + threadIdx.x;
            var stride = gridDim.x * blockDim.x;

            for (var i = start; i < c.Length; i += stride)
            {
                c[i] = a[i] + b[i];
            }
        }

        public static void cuSub(float[] a, float[] b, float[] c)
        {
            var start = blockIdx.x * blockDim.x + threadIdx.x;
            var stride = gridDim.x * blockDim.x;

            for (var i = start; i < c.Length; i += stride)
            {
                c[i] = a[i] - b[i];
            }
        }

        public static void cuConstProd(float[] a, float b, float[] c)
        {
            var start = blockIdx.x * blockDim.x + threadIdx.x;
            var stride = gridDim.x * blockDim.x;

            for (var i = start; i < c.Length; i += stride)
            {
                c[i] = a[i] * b;
            }
        }
        */
    }
}
