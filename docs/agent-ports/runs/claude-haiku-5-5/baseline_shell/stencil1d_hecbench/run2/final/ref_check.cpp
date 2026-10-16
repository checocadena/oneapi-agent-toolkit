#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc,char**argv){
  int n=atoi(argv[2]); const int R=7;
  std::vector<int> a(n+R); for(int i=0;i<n+R;i++) a[i]=i;
  FILE*f=fopen(argv[1],"rb"); std::vector<double> got(n);
  if(fread(got.data(),sizeof(double),n,f)!=(size_t)n){printf("short read\n");return 1;} fclose(f);
  long bad=0;
  for(int g=0;g<n;g++){
    int r=0; for(int o=-R;o<=R;o++){int idx=g+o; if(idx>=0 && idx<n+R) r+=(g<0?0:a[idx]);}
    // reference: out[g] = sum over offsets of tile value; tile value for index<0 is 0, else a[idx]
    r=0; for(int o=-R;o<=R;o++){int idx=g+o; if(idx>=0) r+=a[idx];}
    if(got[g]!=(double)r){ if(bad<5) printf("mismatch g=%d got=%f ref=%d\n",g,got[g],r); bad++; }
  }
  printf("n=%d mismatches=%ld\n",n,bad); return bad!=0;
}
