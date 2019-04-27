#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h> 
#include <omp.h>

#define  n 64

double G   = 1.0;
double c   = 1.0;
double Mpl ;
double H0  = 22.04*(1e-5);


double *psi, *phi, *f,*psi_t, *phi_t, *f_t,*tul00,*tuldss;
double *psik, *phik, *fk,*psik_t, *phik_t, *fk_t,*tul00k,*tuldssk;



double p[n*n*n][6];
double grid[n*n*n];

double fb, fb_a, omdmb, omdeb, a, at, a_t, a_tt, Vamp, ai, a0, da, ak, fbt, fb_a_t, a_t_t;
int jprint;
double H0, Hi;


FILE *fpback;


void background();
double V(double);
double V_f(double);
void initialise();
int evolve(double ,double );

void main()
{       Mpl = 1.0/sqrt(8.0*3.142*G) ;
        da = 0.01;
        jprint = (int) (1.0/da);
         

	fpback  = fopen("back.txt","w");

        int i;

       // i = fftw_init_threads();
	//	fftw_plan_with_nthreads(omp_get_max_threads());

	
	

      

	background();
	initialise();

       i = evolve(ai,a0);
      




}




void background()
{ 
   
   Vamp =1.0;
   int j;
   double Vvl,V_fvl,w,facb,omfb;
   int fail=1;
   ai = 1.0;
   a0 = 1000.0;
   a_t = ai;
   double cpmc = (0.14/(0.68*0.68));
   double ommi = (cpmc)*pow((a0/ai),3.0)/(cpmc*a0*a0*a0/(ai*ai*ai) + (1.0-cpmc));
   double omdei = 1.0-ommi;
   double fb_ak1, fbk1, fb_ak2, fbk2, fb_ak3, fbk3, fb_ak4, fbk4, fbk, fb_ak;
   fb = 1.0;
   fb_a = 0.0/a_t;
    Vamp = 3.0*Mpl*Mpl*omdei*c/V(fb); 
    //printf("Vamp  %.20lf\n",Vamp);
    
 for(j=0,a=ai;a<a0&&(fail>0);a+=da,++j)
    {   Vvl = V(fb);
  	V_fvl = V_f(fb);
	a_t = sqrt((ommi*ai*ai*ai/a  + (1.0/(Mpl*Mpl))*a*a*Vvl/(3.0*c)) / ( 1.0 - (1.0/(Mpl*Mpl))*a*a*fb_a*fb_a/(6.0*c*c*c))) ;
        a_tt = -0.5*ommi*ai*ai*ai/(a*a) - (1.0/(Mpl*Mpl*c))*a*(fb_a*fb_a*a_t*a_t - Vvl)/3.0;
        facb = -V_fvl*c*c/(a_t*a_t) - 3.0*fb_a/a - a_tt*fb_a/(a_t*a_t);
        w = (fb_a*fb_a*a_t*a_t/(2.0*c*c) - Vvl)/(fb_a*fb_a*a_t*a_t/(2.0*c*c) + Vvl);
       
      
      
        ak = a+0.5*da;
        fb_ak1 =  facb*da;
        fbk1 = da*fb_a; 
      
        fb_ak = fb_a + 0.5*fb_ak1;
        fbk = fb + 0.5*fbk1; 
        omfb = (1.0/(3.0*c*c*c*Mpl*Mpl))*(fb_a*fb_a*a_t*a_t/(2.0*c*c) + Vvl); 

        Vvl = V(fbk);
        V_fvl = V_f(fbk);
        a_t = sqrt((ommi*ai*ai*ai/ak  + (1.0/(Mpl*Mpl))*ak*ak*Vvl/(3.0*c)) / ( 1.0 - (1.0/(Mpl*Mpl))*ak*ak*fb_ak*fb_ak/(6.0*c*c*c))) ;
        a_tt = -0.5*ommi*ai*ai*ai/(ak*ak) - (1.0/(Mpl*Mpl*c))*ak*(fb_ak*fb_ak*a_t*a_t - Vvl)/3.0;
        facb = -V_fvl*c*c/(a_t*a_t) - 3.0*fb_ak/ak - a_tt*fb_ak/(a_t*a_t);
      
        ak = a + 0.5*da;
        fb_ak2 = facb*da;
        fbk2 = da*fb_ak; 
      
        fb_ak = fb_a + 0.5*fb_ak2;
        fbk = fb + 0.5*fbk2; 

        Vvl = V(fbk);
    	V_fvl = V_f(fbk);
      	a_t = sqrt((ommi*ai*ai*ai/ak  + (1.0/(Mpl*Mpl))*ak*ak*Vvl/(3.0*c)) / ( 1.0 - (1.0/(Mpl*Mpl))*ak*ak*fb_ak*fb_ak/(6.0*c*c*c))) ;
      	a_tt = -0.5*ommi*ai*ai*ai/(ak*ak) - (1.0/(Mpl*Mpl*c))*ak*(fb_ak*fb_ak*a_t*a_t - Vvl)/3.0;
      	facb = -V_fvl*c*c/(a_t*a_t) - 3.0*fb_ak/ak - a_tt*fb_ak/(a_t*a_t);
      
      	ak = a + da;
      	fb_ak3= facb*da;
      	fbk3 = da*fb_ak; 
      
      	fb_ak = fb_a + fb_ak3;
      	fbk = fb + fbk3; 

        Vvl = V(fbk);
        V_fvl = V_f(fbk);
        a_t = sqrt((ommi*ai*ai*ai/ak  + (1.0/(Mpl*Mpl))*ak*ak*Vvl/(3.0*c)) / ( 1.0 - (1.0/(Mpl*Mpl))*ak*ak*fb_ak*fb_ak/(6.0*c*c*c))) ;
        a_tt = -0.5*ommi*ai*ai*ai/(ak*ak) - (1.0/(Mpl*Mpl*c))*ak*(fb_ak*fb_ak*a_t*a_t - Vvl)/3.0;
        facb = -V_fvl*c*c/(a_t*a_t) - 3.0*fb_ak/ak - a_tt*fb_ak/(a_t*a_t);
      
      
        fb_ak4= facb*da;
        fbk4 = da*fb_ak; 

        fb = fb + (1.0/6.0)*( fbk1 + 2.0*fbk2 + 2.0*fbk3 + fbk4 );
	fb_a = fb_a + (1.0/6.0)*( fb_ak1 + 2.0*fb_ak2 + 2.0*fb_ak3 + fb_ak4 );


         
    }

    a_t = sqrt((ommi*ai*ai*ai/a  + (1.0/(Mpl*Mpl))*a*a*Vvl/(3.0*c)) / ( 1.0 - (1.0/(Mpl*Mpl))*a*a*fb_a*fb_a/(6.0*c*c*c))) ;
    Hi = H0*a/a_t;
    //printf("\nHi    %.20lf  \nratio(Hi/H0)  %.20lf\n",Hi,a/a_t);
    Vvl = V(fb);
    w = (fb_a*fb_a*a_t*a_t/(2.0*c*c) - Vvl)/(fb_a*fb_a*a_t*a_t/(2.0*c*c) + Vvl);

    printf("WWWW  %.10lf  Hi %.10lf\n",w,Hi);
    

  
}


double V(double fff)
{
   return(Vamp*(fff)*(fff));
}


double V_f(double fff)
{
   return(2.0*Vamp*(fff));


}

void initialise()
{
      int ix,iy,iz;

      double cpmc = (0.14/(0.68*0.68));
      a0 = 1000.0;
      ai = 1.0;
      fb = 1.0;
      fb_a = 0.0;
      omdmb= (cpmc)*pow((a0/ai),3.0)/(cpmc*a0*a0*a0/(ai*ai*ai) + (1.0-cpmc));
      
	for(ix = 0;ix <n; ++ix)
	{
		for(iy = 0;iy <n; ++iy)
		{
			for(iz = 0;iz <n; ++iz)
			{
				phi[ix*n*n+iy*n+iz][0] = (double) rand()/((double) RAND_MAX  );
      				phi[ix*n*n+iy*n+iz][1] = 0.0;
				
				psi[ix*n*n+iy*n+iz][0] = (double) rand()/((double) RAND_MAX  );
      				psi[ix*n*n+iy*n+iz][1] = 0.0;
		
				f[ix*n*n+iy*n+iz][0] = 1.0;
      				f[ix*n*n+iy*n+iz][1] = 0.0;

				phi_t[ix*n*n+iy*n+iz][0] = 0.0;
      				phi_t[ix*n*n+iy*n+iz][1] = 0.0;
				
				psi_t[ix*n*n+iy*n+iz][0] = 0.0;
      				psi_t[ix*n*n+iy*n+iz][1] = 0.0;
		
				f_t[ix*n*n+iy*n+iz][0] = 0.0;
      				f_t[ix*n*n+iy*n+iz][1] = 0.0;

				tul00[ix*n*n+iy*n+iz][0] = (double) rand()/((double) RAND_MAX  );
      				tul00[ix*n*n+iy*n+iz][1] = 0.0;

				tuldss[ix*n*n+iy*n+iz][0] = (double) rand()/((double) RAND_MAX  );
      				tuldss[ix*n*n+iy*n+iz][1] = 0.0;

			}

		}
	}
}


int evolve(double aini, double astp)
{
    

    int ix,iy,iz;
    double ommi = omdmb;
    double Vvl,V_fvl,facb;
    double w;

    int fail = 1,i;
    double j,nd = (double) n;
    

    for(a=aini,i=0;(a<=astp)&&(fail==1);a+=da,++i)
	{ if(i%jprint==0)
	   printf("a  %lf\n",a);
          
          Vvl = V(fb);
  	  V_fvl = V_f(fb);
	  a_t = sqrt((ommi*ai*ai*ai/a  + (1.0/(Mpl*Mpl))*a*a*Vvl/(3.0*c)) / ( 1.0 - (1.0/(Mpl*Mpl))*a*a*fb_a*fb_a/(6.0*c*c*c))) ;
          a_tt = -0.5*ommi*ai*ai*ai/(a*a) - (1.0/(Mpl*Mpl*c))*a*(fb_a*fb_a*a_t*a_t - Vvl)/3.0;
          facb = -V_fvl*c*c/(a_t*a_t) - 3.0*fb_a/a - a_tt*fb_a/(a_t*a_t);
	  
	  fbt = fb + 0.5*da*fb_a;
	  fb_a_t = fb_a + 0.5*da*facb;
	  at = a + 0.5*da;

	  fftw_execute(plan_psi_f);
	  fftw_execute(plan_phi_f);
	  fftw_execute(plan_f_f);

           
          #pragma omp parallel for
	  for(j=0.0;j<nd;++j)
	  {
		ix = (int) (floor(j/(nd*nd)));
		iy = (int) (floor(fmod(j,(nd*nd))/nd));
		iz = (int) (j-((double) ix)*nd*nd-((double) iy)*nd);
		


	  }
        
          

	
	  Vvl = V(fbt);
  	  V_fvl = V_f(fbt);
	  a_t = sqrt((ommi*ai*ai*ai/at  + (1.0/(Mpl*Mpl))*at*at*Vvl/(3.0*c)) / ( 1.0 - (1.0/(Mpl*Mpl))*at*at*fb_a_t*fb_a_t/(6.0*c*c*c))) ;
          a_tt = -0.5*ommi*ai*ai*ai/(at*at) - (1.0/(Mpl*Mpl*c))*at*(fb_a_t*fb_a_t*a_t*a_t - Vvl)/3.0;
          facb = -V_fvl*c*c/(a_t*a_t) - 3.0*fb_a_t/at - a_tt*fb_a_t/(a_t*a_t);

          fb = fb + da*fb_a_t;
	  fb_a = fb_a + da*facb;

          if(isnan(fb)||isnan(fb_a))
		{
		  fail = -1;
		  break;
		
		}
	  

         
	}
    a_t = sqrt((ommi*ai*ai*ai/a  + (1.0/(Mpl*Mpl))*a*a*Vvl/(3.0*c)) / ( 1.0 - (1.0/(Mpl*Mpl))*a*a*fb_a*fb_a/(6.0*c*c*c))) ;
    Hi = H0*a/a_t;
    //printf("\nHi    %.20lf  \nratio(Hi/H0)  %.20lf\n",Hi,a/a_t);
    Vvl = V(fb);
    w = (fb_a*fb_a*a_t*a_t/(2.0*c*c) - Vvl)/(fb_a*fb_a*a_t*a_t/(2.0*c*c) + Vvl);

    printf("evolve w  %.10lf  Hi %.10lf  %.10lf  %.10lf\n",w,Hi,fb,a0);
    printf("fail  %d\n",fail);

	return(fail);


}


