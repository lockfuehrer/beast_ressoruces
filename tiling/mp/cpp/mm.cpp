#include <chrono>
#include <cstring>
#include <math.h>
#include <cstdio>
#include <cstdlib>

// use a 1d array to guarantee a dense and sequential memory layout
#define TWO_D_ACCESS(row, col, width) ((width) * (row) + (col))

double mm( long threads, long teams, long N, int REP, double expected_result) {

        double * a = (double *) malloc(N*N*sizeof(double));
        double * b = (double *) malloc(N*N*sizeof(double));
        double * c = (double *) malloc(N*N*sizeof(double));

        #pragma omp target enter data map(alloc: a [ : N * N], b [ : N * N], c [ : N * N])
        #pragma omp target teams distribute parallel for num_teams(teams) num_threads(threads) schedule(static, 1)
        for(int i = 0; i < N*N; ++i){
                a[i] = 0;
                b[i] = atan(i);
                c[i] = cos(i);
        }


        auto t0 = std::chrono::high_resolution_clock::now();

        for( int r=0; r<REP; ++r ) {
                #pragma omp target teams num_teams(teams)
                #pragma omp distribute parallel for collapse(2) num_threads(threads) schedule(static, 1) //dist_schedule(static, 1)
                for( int i=0; i<N; ++i ) {
                        for( int j=0; j<N; ++j) {
                                //double sum = 0.0;
                                //#pragma omp simd reduction(+:sum) simdlen(16)
                                for( int k=0; k<N; ++k ) {
                                        //sum += b[TWO_D_ACCESS(i, k, N)] * c[TWO_D_ACCESS(k, j, N)];
                                        a[TWO_D_ACCESS(i, j, N)] += b[TWO_D_ACCESS(i, k, N)] * c[TWO_D_ACCESS(k, j, N)];
                                }
                                //a[TWO_D_ACCESS(i, k, N)] += sum;
                        }
                }
        }
        #pragma omp target exit data map(delete: b [ : N * N], c [ : N * N]) map(from: a[ : N * N])

        auto t1 = std::chrono::high_resolution_clock::now();

        // simple correctness check
        double array_sum = 0;
        for( int i=0; i<N*N; ++i ) {
                array_sum += a[i];
        }
        // verify expected result. accounting for possibly system dependent float rounding errors
        if(abs(array_sum - expected_result) > 0.001){
                printf("Wrong result for N=%4ld. expected %.3f but got %.3f. Aborting...\n", N, expected_result, array_sum);
                exit(EXIT_FAILURE);
        }
        //printf("Solution(N=%4d): %.3f\n", N, array_sum);

        free((void *) a);
        free((void *) b);
        free((void *) c);
        using dsec = std::chrono::duration<double>;
        double dur = std::chrono::duration_cast<dsec>(t1-t0).count();

        double mflop = 2.0*(double)N*(double)N*(double)N*(double)REP*1.0e-6;
	return mflop/dur;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
//        printf("input errror\n");
//        exit(1);
  }

    char *pEnd;
    long teams=1024;
    long threads=256;
//    long teams= strtol(argv[1], &pEnd, 10);
//    long threads = strtol(argv[2], &pEnd, 10);
 
  // ugly hardcoded stuff
  double mf;

  mf = mm(threads, teams,100, 2000, 219421.205);
  printf("MFLOPS(N=%4d): %.3f\n", 100, mf);

  mf = mm(threads, teams,200, 250, 94647.218);
  printf("MFLOPS(N=%4d): %.3f\n", 200, mf);

  mf = mm(threads, teams,300, 75, -9954.384);
  printf("MFLOPS(N=%4d): %.3f\n", 300, mf);

  mf = mm(threads, teams,400, 32, -10319.503);
  printf("MFLOPS(N=%4d): %.3f\n", 400, mf);

  mf = mm(threads, teams,500, 16, -4625.229);
  printf("MFLOPS(N=%4d): %.3f\n", 600, mf);

  mf = mm(threads, teams,600, 10, -4645.287);
  printf("MFLOPS(N=%4d): %.3f\n", 600, mf);

  mf = mm(threads, teams,700, 6, -2458.730);
  printf("MFLOPS(N=%4d): %.3f\n", 700, mf);

  mf = mm(threads, teams,800, 4, 5147.686);
  printf("MFLOPS(N=%4d): %.3f\n", 800, mf);

  mf = mm(threads, teams,900, 3, 4140.630);
  printf("MFLOPS(N=%4d): %.3f\n", 900, mf);

  mf = mm(threads, teams,1000, 2, -910.024);
  printf("MFLOPS(N=%4d): %.3f\n", 1000, mf);

  mf = mm(threads, teams,1100, 2, 3815.298);
  printf("MFLOPS(N=%4d): %.3f\n", 1100, mf);

  mf = mm(threads, teams,1300, 1, -1083.652);
  printf("MFLOPS(N=%4d): %.3f\n", 1300, mf);

  mf = mm(threads, teams,1500, 1, 531.104);
  printf("MFLOPS(N=%4d): %.3f\n", 1500, mf);

  mf = mm(threads, teams,1700, 1, -1344.734);
  printf("MFLOPS(N=%4d): %.3f\n", 1700, mf);

  mf = mm(threads, teams,1900, 1, 4589.767);
  printf("MFLOPS(N=%4d): %.3f\n", 1900, mf);
  return 0;
}
