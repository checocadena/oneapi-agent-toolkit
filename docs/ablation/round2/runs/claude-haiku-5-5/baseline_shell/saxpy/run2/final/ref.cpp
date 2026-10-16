#include <cstdio>
#include <cstdlib>
#include <vector>
int main(int argc,char**argv){int n=argc>2?atoi(argv[2]):(1<<20);const float a=2.5f;
std::vector<float>x(n),y(n);for(int i=0;i<n;++i){x[i]=static_cast<float>(i%1000)*0.001f;y[i]=static_cast<float>((i*7)%1000)*0.002f;}
for(int i=0;i<n;++i)y[i]=a*x[i]+y[i];
FILE*f=fopen(argv[1],"wb");fwrite(y.data(),4,n,f);fclose(f);}
