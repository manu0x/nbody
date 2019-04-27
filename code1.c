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


double *psi, *phi, *f,*psi_a, *phi_a, *f_a,*tul00,*tuldss;
double *tmppsi, *tmpphi, *tmpf,*tmppsi_a, *tmpphi_a, *tmpf_a,*tmptul00,*tmptuldss;
double dx,dy,dz;




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

	psi = (double *) malloc(n*n*n*sizeof(double)); 
        psi_a = (double *) malloc(n*n*n*sizeof(double)); 
	phi = (double *) malloc(n*n*n*sizeof(double)); 
        phi_a = (double *) malloc(n*n*n*sizeof(double)); 
	f = (double *) malloc(n*n*n*sizeof(double)); 
        f_a = (double *) malloc(n*n*n*sizeof(double)); 
	tul00 = (double *) malloc(n*n*n*sizeof(double)); 
        tuldss = (double *) malloc(n*n*n*sizeof(double));

	tmppsi = (double *) malloc(n*n*n*sizeof(double)); 
        tmppsi_a = (double *) malloc(n*n*n*sizeof(double)); 
	tmpphi = (double *) malloc(n*n*n*sizeof(double)); 
        tmpphi_a = (double *) malloc(n*n*n*sizeof(double)); 
	tmpf = (double *) malloc(n*n*n*sizeof(double)); 
        tmpf_a = (double *) malloc(n*n*n*sizeof(double)); 
	tmptul00 = (double *) malloc(n*n*n*sizeof(double)); 
        tmptuldss = (double *) malloc(n*n*n*sizeof(double)); 
	

      

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

	dx = 0.001; dy =0.001; dz = 0.001;
      
	for(ix = 0;ix <n; ++ix)
	{
		for(iy = 0;iy <n; ++iy)
		{
			for(iz = 0;iz <n; ++iz)
			{
				phi[ix*n*n+iy*n+iz] = (double) rand()/((double) RAND_MAX  );
      				
				
				psi[ix*n*n+iy*n+iz] = (double) rand()/((double) RAND_MAX  );
      				
		
				f[ix*n*n+iy*n+iz] = 1.0;
      				

				phi_t[ix*n*n+iy*n+iz] = 0.0;
      				
				
				psi_t[ix*n*n+iy*n+iz] = 0.0;
      				
		
				f_t[ix*n*n+iy*n+iz] = 0.0;
      				

				tul00[ix*n*n+iy*n+iz] = (double) rand()/((double) RAND_MAX  );
      				

				tuldss[ix*n*n+iy*n+iz] = (double) rand()/((double) RAND_MAX  );
      				

			}

		}
	}
}


int evolve(double aini, double astp)
{
    


    double ommi = omdmb;
    double facb;
    double w;

    int fail = 1,i,j;

    double nd = (double) n, jd;  ///Watch out for local vs global for parallelization
    double Phiacc,facc,Vvl,V_fvl;
    int ix,iy,iz,ci,cnxp1,cnxp2,cnxn1,cnxn2,cnyp1,cnyp2,cnyn1,cnyn2,cnzp1,cnzp2,cnzn1,cnzn2;
    double lplPhi,lplPsi,lplf;
    

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

	 

           
          #pragma omp parallel for
	  for(j=0;j<n;++j)
	  {     jd = j;
		ix = (int) (floor(jd/(nd*nd)));
		iy = (int) (floor(fmod(jd,(nd*nd))/nd));
		iz = (int) (jd-((double) ix)*nd*nd-((double) iy)*nd);
		ci = ix*n*n+iy*n+iz;
		cnxp1 = (n + (ix+1))%n;
		cnxp2 = (n + (ix+2))%n;
		cnxn1 = (n + (ix-1))%n;
		cnxn2 = (n + (ix-2))%n;
		cnyp1 = (n + (iy+1))%n;
		cnyp2 = (n + (iy+2))%n;
		cnyn1 = (n + (iy-1))%n;
		cnyn2 = (n + (iy-2))%n;
		cnzp1 = (n + (iz+1))%n;
		cnzp2 = (n + (iz+2))%n;
		cnzn1 = (n + (iz-1))%n;
		cnzn2 = (n + (iz-2))%n;

		lplPhi = (-Phi[cnxn2]-16.0*Phi[cnxn1]-30.0*Phi[ci]+16.0*Phi[cnxnp1]-Phi[cnxp2])/(12.0*dx*dx) +
			 (-Phi[cnyn2]-16.0*Phi[cnyn1]-30.0*Phi[ci]+16.0*Phi[cnynp1]-Phi[cnyp2])/(12.0*dy*dy) +
			 (-Phi[cnzn2]-16.0*Phi[cnzn1]-30.0*Phi[ci]+16.0*Phi[cnznp1]-Phi[cnzp2])/(12.0*dz*dz)   ;

		lplPsi = (-Psi[cnxn2]-16.0*Psi[cnxn1]-30.0*Psi[ci]+16.0*Psi[cnxnp1]-Psi[cnxp2])/(12.0*dx*dx) +
			 (-Psi[cnyn2]-16.0*Psi[cnyn1]-30.0*Psi[ci]+16.0*Psi[cnynp1]-Psi[cnyp2])/(12.0*dy*dy) +
			 (-Psi[cnzn2]-16.0*Psi[cnzn1]-30.0*Psi[ci]+16.0*Psi[cnznp1]-Psi[cnzp2])/(12.0*dz*dz)   ;

		lplf = (-f[cnxn2]-16.0*f[cnxn1]-30.0*f[ci]+16.0*f[cnxnp1]-f[cnxp2])/(12.0*dx*dx) +
			 (-f[cnyn2]-16.0*f[cnyn1]-30.0*f[ci]+16.0*f[cnynp1]-f[cnyp2])/(12.0*dy*dy) +
			 (-f[cnzn2]-16.0*f[cnzn1]-30.0*f[ci]+16.0*f[cnznp1]-f[cnzp2])/(12.0*dz*dz)   ;

		Vvl = V(f[ci]);
  	  	V_fvl = V_f(f[ci]);
	
		Phiacc = (0.5/(a*a))*( -2.0*Psi[ci] - 4.0*a*Psi[ci]*a_tt/(a_t*a_t) - 6.0*a*Phi_a[ci]- 2.0*a*Psi_a[ci]  + 
			               (2.0/3.0)*(lplPhi-lplPsi)/(a_t*a_t) - a*a*tuldss/(Mpl*3.0*(a_t*a_t)) - Phi_a[ci]*a_tt/(a_t*a_t)
                                      );

		facc = (1.0/(a*a*(-1.0+2.0*Psi[ci]))) *(  ( a*a*V_fvl 

                                                           + (f[cnxn2] - 8.0*f[cnxn1] + 8.0*f[cnxp1] - f[cnxp2])*
							     ((Phi[cnxn2]-Psi[cnxn2]) - 8.0*(Phi[cnxn1]-Psi[cnxn1])
                                                               + 8.0*(Phi[cnxp1]-Psi[cnxp1]) - (Phi[cnxp2]-Psi[cnxp2]))/(dx*dx)

							       + (f[cnyn2] - 8.0*f[cnyn1] + 8.0*f[cnyp1] - f[cnyp2])*
							     ((Phi[cnyn2]-Psi[cnyn2]) - 8.0*(Phi[cnyn1]-Psi[cnyn1])
                                                               + 8.0*(Phi[cnyp1]-Psi[cnyp1]) - (Phi[cnyp2]-Psi[cnyp2]))/(dy*dy)

							       + (f[cnzn2] - 8.0*f[cnzn1] + 8.0*f[cnzp1] - f[cnzp2])*
							     ((Phi[cnzn2]-Psi[cnzn2]) - 8.0*(Phi[cnzn1]-Psi[cnzn1])
                                                               + 8.0*(Phi[cnzp1]-Psi[cnzp1]) - (Phi[cnzp2]-Psi[cnzp2]))/(dz*dz)  
				
							   -lplf*(1.0+2.0*Phi[ci])	- a_tt*f_a[ci]
                                                         )/(a_t*a_t)

		                                        +   ( 3.0*a*f_a[ci] - 3.0*a*a*Phi_a[ci]*f_a[ci] 
                                                          -6.0*a*Psi[ci]*f_a[ci] - a*a*f_a[ci]*Psi[ci] )
						      );

                tmpPhi_a[ci] = Phi_a[ci] + 0.5*Phiacc*da;
		tmpf_a[ci] = f_a[ci] + 0.5*facc*da;

		tmpPhi[ci] = Phi[ci] + 0.5*Phi_a*da;
		tmpf[ci] = f[ci] + 0.5*f_a*da;

		tmpPsi[ci] = 
		

		

		++jd;
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


