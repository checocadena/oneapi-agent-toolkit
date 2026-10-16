#include <vector>
#include <cstdio>
#include <cstdlib>
int main(int argc,char**argv){
  int n=atoi(argv[2]); int repeat=atoi(argv[3]);
  std::vector<float> m(size_t(n)*n);
  for(int i=0;i<n;i++)for(int j=0;j<n;j++) m[size_t(i)*n+j]=i*n+j;
  for(int r=0;r<repeat;r++)
    for(int layer=0;layer<n/2;layer++){
      int first=layer,last=n-1-layer;
      for(int i=first;i<last;++i){
        int offset=i-first;
        float top=m[first*n+i];
        m[first*n+i]=m[(last-offset)*n+first];
        m[(last-offset)*n+first]=m[last*n+(last-offset)];
        m[last*n+(last-offset)]=m[i*n+last];
        m[i*n+last]=top;
      }
    }
  FILE*f=fopen(argv[1],"wb"); fwrite(m.data(),sizeof(float),m.size(),f); fclose(f);
}
