#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <chrono>
#include <algorithm>

#include <omp.h>

double calculateMegaFlopRate(long size, long repetitions, double duration) {
        long numOfMegaFlops = (repetitions * (size * 2))/1000000;
        return (double) numOfMegaFlops/duration;
}

double calculateChecksum(long datasetSize, const volatile double* vector) {
        double checksum = 0;
        for (int i = 0; i < datasetSize; i++) {
                checksum += vector[i];
        }
        return checksum;
}

void triad(long datasetSize, long repetitions, long numThreads, long numTeams) {

        volatile double *A, *B, *C, *D;

        A = (double*) malloc(datasetSize * sizeof(double));
        B = (double*) malloc(datasetSize * sizeof(double));
        C = (double*) malloc(datasetSize * sizeof(double));
        D = (double*) malloc(datasetSize * sizeof(double));

        for (int i = 0; i < datasetSize; i++) {
                A[i] = i;
                B[i] = i;
                C[i] = i;
                D[i] = i;
        }

        auto start_time = std::chrono::high_resolution_clock::now();

        #pragma omp target enter data map (to: C[ :datasetSize], B[ :datasetSize], D[ :datasetSize], A[ :datasetSize])
        #pragma omp target teams num_teams(numTeams)
        #pragma omp distribute parallel for collapse(2)
        for (int i = 0; i < repetitions; i++) {
                for (int j = 0; j < datasetSize; j++) {
                        A[j] = B[j] + C[j] * D[j];
                }
        }

        auto end_time = std::chrono::high_resolution_clock::now();

        #pragma omp target exit data map(delete: C[ :datasetSize], B[ :datasetSize], D[ :datasetSize]) map(from: A[ :datasetSize])


        std::chrono::duration<double> elapsed_seconds = end_time-start_time;
        double duration = elapsed_seconds.count();
        double checksum = calculateChecksum(datasetSize, A);

        double mflops = calculateMegaFlopRate(datasetSize, repetitions, duration);
        printf("| %10ld | %8ld | %8ld | %8.2f | %8ld | %.4e |\n", datasetSize, numTeams, numThreads, mflops, repetitions, checksum);

        free((void*)A);
        free((void*)B);
        free((void*)C);
        free((void*)D);
}

int main(int argc, char *argv[]) {

        long maximumDatasetSize =  134217728;
        long totalNumberProcessedPoints = 268435456;

        fprintf(stderr, "Maximum dataset size = %ld, total number of processed points = %ld. Performance in MFLOPS.\n", maximumDatasetSize, totalNumberProcessedPoints);
        printf("| %10s | %8s | %8s | %8s | %8s | %10s |\n", "Data size", "Teams", "Threads", "MFLOPS", "Cycles", "Checksum");

        long datasetSize = 64;
        long t[11] = {1,2,4,8,16,32,64,128,256,512,1024};
        long T[11] = {1024,512,246,128,64,32,16,8,4,2,1};
        while (datasetSize <= maximumDatasetSize) {
 		long cycles = std::clamp(totalNumberProcessedPoints / datasetSize, 8l, 65536l);
#ifdef _OPENMP
                long num_threads = omp_get_max_threads();
#else
                long num_threads = 1;
#endif
                //for (long threads = num_threads; threads > 0; threads /= 2) {
                //      triad(datasetSize, cycles, threads);
                //}
                //for (int i = 0; i < 11; i++){
                //        triad(datasetSize, cycles, t[i], T[i]);
                //}
                triad(datasetSize, cycles, 1024, 256);

                datasetSize *= 2;
        }

        return 0;
}
