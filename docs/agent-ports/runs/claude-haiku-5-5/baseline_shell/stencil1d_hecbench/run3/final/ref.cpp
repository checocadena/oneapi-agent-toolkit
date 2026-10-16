#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc,char**argv){
  int length = atoi(argv[2]); const int R=7;
  std::vector<int> a(length+R); for(int i=0;i<length+R;i++) a[i]=i;
  FILE*f=fopen(argv[1],"wb");
  for(int g=0;g<length;g++){
    int r=0;
    for(int o=-R;o<=R;o++){ int j=g+o; r += (j<0)?0:a[j]; }
    double d=r; fwrite(&d,sizeof(double),1,f);
  }
  fclose(f); return 0;
}
