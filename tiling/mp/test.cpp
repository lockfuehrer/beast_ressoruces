int main(int argc, char* argv[]){
  double a*;
  a = {10,20,30};
#pragma omp target map(to: a[0:2])
{
  printf("index0: %d", a);
}

}
