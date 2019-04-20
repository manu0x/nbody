#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h> 

#define  n 512 

void main()
{  int i,j,k;
  
  FILE *fps = fopen("sig.txt","w");
  FILE *fpf = fopen("fsig.txt","w");
  FILE *fpsi = fopen("isig.txt","w");
  printf("YoYo\n");

  fftw_complex *in, *out;
  fftw_plan p;
  
  in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n*n);
  out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n*n);

  p = fftw_plan_dft_2d(n,n, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

  for(i=0;i<n;++i)
  {  for(j=0;j<n;++j)
     {
      in[i*n+j][0] = (double) rand()/((double) RAND_MAX  );
      in[i*n+j][1] = 0.0;
     }




   }

    
  
   fftw_execute(p);

  for(i=0;i<n;++i)
  { for(j=0;j<n;++j)
     { fprintf(fps,"%lf\t%lf\t%.10lf\t%.10lf\n",(double) i,(double) j,in[i*n+j][0],in[i*n+j][1]);
    
 
       fprintf(fpf,"%lf\t%lf\t%.10lf\t%.10lf\n",(double) i,(double) j,out[i*n+j][0],out[i*n+j][1]);
     }
  }


  fftw_destroy_plan(p); 

  p = fftw_plan_dft_2d(n,n, out, in, FFTW_BACKWARD, FFTW_ESTIMATE);
     fftw_execute(p);


 for(i=0;i<n;++i)
  { for(j=0;j<n;++j)
     { 
       fprintf(fpsi,"%lf\t%lf\t%.10lf\t%.10lf\n",(double) i,(double) j,in[i*n+j][0],in[i*n+j][1]);
       
     }
  }




}
