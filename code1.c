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
double L;
int N;


double *psi, *phi, *f,*psi_a, *phi_a, *f_a,*tul00,*tuldss;
double *tmppsi, *tmpphi, *tmpf,*tmppsi_a, *tmpphi_a, *tmpf_a,*tmptul00,*tmptuldss,*m;
double dx,dy,dz;




double p[n*n*n][6];
double grid[n*n*n];

int nic[n*n*n][16];

double fb, fb_a, omdmb, omdeb, a, at, a_t, a_tt, Vamp, ai, a0, da, ak, fbt, fb_a_t, a_t_t;
int jprint;
double H0, Hi;


FILE *fpback;


void background();
void neighindc();
double V(double);
double V_f(double);
void initialise();
int evolve(double ,double );

void main()
{       Mpl = 1.0/sqrt(8.0*3.142*G) ;
        da = 0.01;
        jprint = (int) (1.0/da);
	N=n*n*n;
         

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

	m = (double *) malloc(n*n*n*sizeof(double)); 
	

      

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


void neighindc()
{
	int i,ix,iy,iz;
	double jd,nd;
        nd = (double) nd;
	
	

	for(i=0;i<n;++i)
	{
		               
		
		
		nic[i][0]  = (N+i-(n*n))%N;
		nic[i][1]  = (N+i-(n))%N;
		nic[i][2]  = (N+i-1)%N;
		nic[i][3]  = (N+i+(n*n))%N;
		nic[i][4]  = (N+i+(n))%N;
		nic[i][5]  = (N+i+1)%N;

		nic[i][6]  = (N+i-2*(n*n))%N;
		nic[i][7]  = (N+i-2*(n))%N;
		nic[i][8]  = (N+i-2)%N;
		nic[i][9]  = (N+i+2*(n*n))%N;
		nic[i][10]  = (N+i+2*(n))%N;
		nic[i][11]  = (N+i+2)%N;

		nic[i][12] = (i+(n*n)+n)%N
		nic[i][13] = (i+n+1)%N
		nic[i][14] = (i+(n*n)+1)%N
		nic[i][15] = (N+i+(n*n)+n+1)%N
		


         }




}


void initialise()
{
      int ix,iy,iz,Vvl,V_fvl;

      double cpmc = (0.14/(0.68*0.68));
      int px,py,pz,ci,pci;
      double gamma, v, gradmgf;
      a0 = 1000.0;
      ai = 1.0;
      fb = 1.0;
      fb_a = 0.0;
      omdmb= (cpmc)*pow((a0/ai),3.0)/(cpmc*a0*a0*a0/(ai*ai*ai) + (1.0-cpmc));

	dx = 0.001; dy =0.001; dz = 0.001;
        L = dx*((double) n); 
      
	for(ix = 0;ix <n; ++ix)
	{
		for(iy = 0;iy <n; ++iy)
		{
			for(iz = 0;iz <n; ++iz)
			{       ci = ix*n*n+iy*n+iz;
				phi[ix*n*n+iy*n+iz] = (double) rand()/((double) RAND_MAX  );
      				
				
				psi[ix*n*n+iy*n+iz] = (double) rand()/((double) RAND_MAX  );
      				
		
				f[ix*n*n+iy*n+iz] = fb;
      				

				phi_a[ix*n*n+iy*n+iz] = 0.0;
      				
				
				psi_a[ix*n*n+iy*n+iz] = 0.0;
      				
		
				f_a[ix*n*n+iy*n+iz] = fb_a;

				p[ix*n*n+iy*n+iz][0] =  ((double) rand()/((double) RAND_MAX  ))*L;
				px = (int)(p[ix*n*n+iy*n+iz][0]/dx);
				p[ix*n*n+iy*n+iz][1] =  ((double) rand()/((double) RAND_MAX  ))*L;
				py = (int)(p[ix*n*n+iy*n+iz][1]/dy);
				p[ix*n*n+iy*n+iz][2] =  ((double) rand()/((double) RAND_MAX  ))*L;
				pz = (int)(p[ix*n*n+iy*n+iz][3]/dx);

				pci = px*n*n+py*n+pz;

				p[ix*n*n+iy*n+iz][4] =  0.0;
				p[ix*n*n+iy*n+iz][5] =  0.0;
				p[ix*n*n+iy*n+iz][6] =  0.0;

				v = sqrt(p[ix*n*n+iy*n+iz][3]*p[ix*n*n+iy*n+iz][3] + p[ix*n*n+iy*n+iz][4]*p[ix*n*n+iy*n+iz][4]
                                         + p[ix*n*n+iy*n+iz][5]*p[ix*n*n+iy*n+iz][5]);
				gamma = 1.0/sqrt(1.0-a*a*v*v);


//////////////////////////////////particle tul calculation begins/////////////////////////////////////////////////////////////////////////////////

				
      				tul00[px*n*n+py*n+pz]+= (1.0/(8.0))* m[ix*n*n+iy*n+iz]*gamma*(1.0 + 3.0*phi[px*n*n+py*n+pz] - psi[px*n*n+py*n+pz]  
                                                            -gamma*gamma*(v*v*a*a*phi[px*n*n+py*n+pz] + psi[px*n*n+py*n+pz])  );
		tul00[(px+1)*n*n+py*n+pz]+=(1.0/(8.0))*  m[ix*n*n+iy*n+iz]*gamma*(1.0 + 3.0*phi[(px+1)*n*n+py*n+pz] - psi[(px+1)*n*n+py*n+pz]  
                                                            -gamma*gamma*(v*v*a*a*phi[(px+1)*n*n+py*n+pz] + psi[(px+1)*n*n+py*n+pz])  );
		tul00[px*n*n+(py+1)*n+pz]+=(1.0/(8.0))*  m[ix*n*n+iy*n+iz]*gamma*(1.0 + 3.0*phi[px*n*n+(py+1)*n+pz] - psi[px*n*n+(py+1)*n+pz]  
                                                            -gamma*gamma*(v*v*a*a*phi[px*n*n+(py+1)*n+pz] + psi[px*n*n+(py+1)*n+pz])  );
		tul00[px*n*n+py*n+(pz+1)]+=  (1.0/(8.0))*m[ix*n*n+iy*n+iz]*gamma*(1.0 + 3.0*phi[px*n*n+py*n+(pz+1)] - psi[px*n*n+py*n+(pz+1)]  
                                                            -gamma*gamma*(v*v*a*a*phi[px*n*n+py*n+(pz+1)] + psi[px*n*n+py*n+(pz+1)])  );
		tul00[(px+1)*n*n+(py+1)*n+pz]+= 
                                      (1.0/(8.0))* m[ix*n*n+iy*n+iz]*gamma*(1.0 + 3.0*phi[(px+1)*n*n+(py+1)*n+pz] - psi[(px+1)*n*n+(py+1)*n+pz]  
                                                            -gamma*gamma*(v*v*a*a*phi[(px+1)*n*n+(py+1)*n+pz] + psi[(px+1)*n*n+(py+1)*n+pz])  );
				tul00[(px+1)*n*n+py*n+(pz+1)]+= 
					(1.0/(8.0))*m[ix*n*n+iy*n+iz]*gamma*(1.0 + 3.0*phi[(px+1)*n*n+py*n+(pz+1)] - psi[(px+1)*n*n+py*n+(pz+1)]  
                                                            -gamma*gamma*(v*v*a*a*phi[(px+1)*n*n+py*n+(pz+1)] + psi[(px+1)*n*n+py*n+(pz+1)])  );
				tul00[px*n*n+(py+1)*n+(pz+1)]+=  
					(1.0/(8.0))*m[ix*n*n+iy*n+iz]*gamma*(1.0 + 3.0*phi[px*n*n+(py+1)*n+(pz+1)] - psi[px*n*n+(py+1)*n+(pz+1)]  
                                                            -gamma*gamma*(v*v*a*a*phi[px*n*n+(py+1)*n+(pz+1)] + psi[px*n*n+(py+1)*n+(pz+1)])  );
				tul00[(px+1)*n*n+(py+1)*n+(pz+1)]+=  
				(1.0/(8.0))*m[ix*n*n+iy*n+iz]*gamma*(1.0 + 3.0*phi[(px+1)*n*n+(py+1)*n+(pz+1)] - psi[(px+1)*n*n+(py+1)*n+(pz+1)]  
                                                    -gamma*gamma*(v*v*a*a*phi[(px+1)*n*n+(py+1)*n+(pz+1)] + psi[(px+1)*n*n+(py+1)*n+(pz+1)])  );
//////////////////////////////////////diagonal sum of spatial tu//////////////////////////////////////////////////////////////////////////////////
				tuldss[px*n*n+py*n+pz]+= (1.0/(24.0))*m[ix*n*n+iy*n+iz]*gamma*v*v*
								(1.0 + 3.0*phi[px*n*n+py*n+pz] - psi[px*n*n+py*n+pz]  
                                                            -gamma*v*v*gamma*v*v*(v*v*a*a*phi[px*n*n+py*n+pz] + psi[px*n*n+py*n+pz])  );
				tuldss[(px+1)*n*n+py*n+pz]+= 
						         (1.0/(24.0))*m[ix*n*n+iy*n+iz]*gamma*v*v*
									(1.0 + 3.0*phi[(px+1)*n*n+py*n+pz] - psi[(px+1)*n*n+py*n+pz]  
                                                           -gamma*v*v*gamma*v*v*(v*v*a*a*phi[(px+1)*n*n+py*n+pz] + psi[(px+1)*n*n+py*n+pz])  );
				tuldss[px*n*n+(py+1)*n+pz]+=  
							 (1.0/(24.0))*m[ix*n*n+iy*n+iz]*gamma*v*v*
									(1.0 + 3.0*phi[px*n*n+(py+1)*n+pz] - psi[px*n*n+(py+1)*n+pz]  
                                                             -gamma*v*v*gamma*v*v*(v*v*a*a*phi[px*n*n+(py+1)*n+pz] + psi[px*n*n+(py+1)*n+pz])  );
				tuldss[px*n*n+py*n+(pz+1)]+=  
							(1.0/(24.0))*m[ix*n*n+iy*n+iz]*gamma*v*v*
									(1.0 + 3.0*phi[px*n*n+py*n+(pz+1)] - psi[px*n*n+py*n+(pz+1)]  
                                                            -gamma*v*v*gamma*v*v*(v*v*a*a*phi[px*n*n+py*n+(pz+1)] + psi[px*n*n+py*n+(pz+1)])  );
				tuldss[(px+1)*n*n+(py+1)*n+pz]+= 
                                                 (1.0/(24.0))*m[ix*n*n+iy*n+iz]*gamma*v*v*
								(1.0 + 3.0*phi[(px+1)*n*n+(py+1)*n+pz] - psi[(px+1)*n*n+(py+1)*n+pz]  
                                                       -gamma*v*v*gamma*v*v*(v*v*a*a*phi[(px+1)*n*n+(py+1)*n+pz] + psi[(px+1)*n*n+(py+1)*n+pz])  );
				tuldss[(px+1)*n*n+py*n+(pz+1)]+= 
						 (1.0/(24.0))*m[ix*n*n+iy*n+iz]*gamma*v*v*
									(1.0 + 3.0*phi[(px+1)*n*n+py*n+(pz+1)] - psi[(px+1)*n*n+py*n+(pz+1)]  
                                                       -gamma*v*v*gamma*v*v*(v*v*a*a*phi[(px+1)*n*n+py*n+(pz+1)] + psi[(px+1)*n*n+py*n+(pz+1)])  );
				tuldss[px*n*n+(py+1)*n+(pz+1)]+=  
						 (1.0/(24.0))*m[ix*n*n+iy*n+iz]*gamma*v*v*
								(1.0 + 3.0*phi[px*n*n+(py+1)*n+(pz+1)] - psi[px*n*n+(py+1)*n+(pz+1)]  
                                                       -gamma*v*v*gamma*v*v*(v*v*a*a*phi[px*n*n+(py+1)*n+(pz+1)] + psi[px*n*n+(py+1)*n+(pz+1)])  );
				tuldss[(px+1)*n*n+(py+1)*n+(pz+1)]+=  
					(1.0/(18.0))*m[ix*n*n+iy*n+iz]*gamma*v*v*
						(1.0 + 3.0*phi[(px+1)*n*n+(py+1)*n+(pz+1)] - psi[(px+1)*n*n+(py+1)*n+(pz+1)]  
                                              -gamma*v*v*gamma*v*v*(v*v*a*a*phi[(px+1)*n*n+(py+1)*n+(pz+1)] + psi[(px+1)*n*n+(py+1)*n+(pz+1)])  );
///////////////////////////////////particle tul calculation end/////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////Quint field tul cal begins with tul00/////////////////////////////////////////////////////////////////////
				Vvl = V(f[ix*n*n+iy*n+iz]);
				grdmagf =  ((f[cnxn2] - 8.0*f[cnxn1] + 8.0*f[cnxp1] - f[cnxp2])*
							    /(12.0*dx*dx))*((f[cnxn2] - 8.0*f[cnxn1] + 8.0*f[cnxp1] - f[cnxp2])*
							    /(12.0*dx*dx))
					   + ((f[cnyn2] - 8.0*f[cnyn1] + 8.0*f[cnyp1] - f[cnyp2])*
							    /(12.0*dy*dy))*((f[cnyn2] - 8.0*f[cnyn1] + 8.0*f[cnyp1] - f[cnyp2])*
							    /(12.0*dy*dy))
					  + ((f[cnzn2] - 8.0*f[cnzn1] + 8.0*f[cnzp1] - f[cnzp2])*
							    /(12.0*dz*dz))*((f[cnzn2] - 8.0*f[cnzn1] + 8.0*f[cnzp1] - f[cnzp2])*
							    /(12.0*dz*dz));



							       + (f[cnyn2] - 8.0*f[cnyn1] + 8.0*f[cnyp1] - f[cnyp2])*
							     ((phi[cnyn2]-psi[cnyn2]) - 8.0*(phi[cnyn1]-psi[cnyn1])
                                                               + 8.0*(phi[cnyp1]-psi[cnyp1]) - (phi[cnyp2]-psi[cnyp2]))/(12.0*dy*dy)

							       + (f[cnzn2] - 8.0*f[cnzn1] + 8.0*f[cnzp1] - f[cnzp2])*
							     ((phi[cnzn2]-psi[cnzn2]) - 8.0*(phi[cnzn1]-psi[cnzn1])
                                                               + 8.0*(phi[cnzp1]-psi[cnzp1]) - (phi[cnzp2]-psi[cnzp2]))/(12.0*dz*dz) 

				tul00[ix*n*n+iy*n+iz]  = Vvl + 0.5*fb_a[ix*n*n+iy*n+iz]*fb_a[ix*n*n+iy*n+iz]*a_t*a_t*(1.0-2.0*Psi[ix*n*n+iy*n+iz])
							  + 0.5*grdmagf*(1.0+2.0*Phi[ix*n*n+iy*n+iz])/(a*a);
				tuldss[ix*n*n+iy*n+iz] = Vvl - 0.5*fb_a[ix*n*n+iy*n+iz]*fb_a[ix*n*n+iy*n+iz]*a_t*a_t*(1.0-2.0*Psi[ix*n*n+iy*n+iz])
							  + grdmagf*(1.0+2.0*Phi[ix*n*n+iy*n+iz])/(6.0*a*a);	
		

				
				
	
      				

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
    double phiacc,facc,Vvl,V_fvl;
    int ix,iy,iz,ci,cnxp1,cnxp2,cnxn1,cnxn2,cnyp1,cnyp2,cnyn1,cnyn2,cnzp1,cnzp2,cnzn1,cnzn2;
    double lplphi,lplpsi,lplf;
    

    for(a=aini,i=0;(a<=astp)&&(fail==1);++i)
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

		lplphi = (-phi[cnxn2]+16.0*phi[cnxn1]-30.0*phi[ci]+16.0*phi[cnxp1]-phi[cnxp2])/(12.0*dx*dx) +
			 (-phi[cnyn2]+16.0*phi[cnyn1]-30.0*phi[ci]+16.0*phi[cnyp1]-phi[cnyp2])/(12.0*dy*dy) +
			 (-phi[cnzn2]+16.0*phi[cnzn1]-30.0*phi[ci]+16.0*phi[cnzp1]-phi[cnzp2])/(12.0*dz*dz)   ;

		lplpsi = (-psi[cnxn2]+16.0*psi[cnxn1]-30.0*psi[ci]+16.0*psi[cnxp1]-psi[cnxp2])/(12.0*dx*dx) +
			 (-psi[cnyn2]+16.0*psi[cnyn1]-30.0*psi[ci]+16.0*psi[cnyp1]-psi[cnyp2])/(12.0*dy*dy) +
			 (-psi[cnzn2]+16.0*psi[cnzn1]-30.0*psi[ci]+16.0*psi[cnzp1]-psi[cnzp2])/(12.0*dz*dz)   ;

		lplf = (-f[cnxn2]+16.0*f[cnxn1]-30.0*f[ci]+16.0*f[cnxp1]-f[cnxp2])/(12.0*dx*dx) +
			 (-f[cnyn2]+16.0*f[cnyn1]-30.0*f[ci]+16.0*f[cnyp1]-f[cnyp2])/(12.0*dy*dy) +
			 (-f[cnzn2]+16.0*f[cnzn1]-30.0*f[ci]+16.0*f[cnzp1]-f[cnzp2])/(12.0*dz*dz)   ;

		Vvl = V(f[ci]);
  	  	V_fvl = V_f(f[ci]);
	
		phiacc = (0.5/(a*a))*( -2.0*psi[ci] - 4.0*a*psi[ci]*a_tt/(a_t*a_t) - 6.0*a*phi_a[ci]- 2.0*a*psi_a[ci]  + 
			               (2.0/3.0)*(lplphi-lplpsi)/(a_t*a_t) - a*a*tuldss[ci]/(Mpl*3.0*(a_t*a_t)) - phi_a[ci]*a_tt/(a_t*a_t)
                                      );

		facc = (1.0/(a*a*(-1.0+2.0*psi[ci]))) *(  ( a*a*V_fvl 

                                                           + (f[cnxn2] - 8.0*f[cnxn1] + 8.0*f[cnxp1] - f[cnxp2])*
							     ((phi[cnxn2]-psi[cnxn2]) - 8.0*(phi[cnxn1]-psi[cnxn1])
                                                               + 8.0*(phi[cnxp1]-psi[cnxp1]) - (phi[cnxp2]-psi[cnxp2]))/(12.0*dx*dx)

							       + (f[cnyn2] - 8.0*f[cnyn1] + 8.0*f[cnyp1] - f[cnyp2])*
							     ((phi[cnyn2]-psi[cnyn2]) - 8.0*(phi[cnyn1]-psi[cnyn1])
                                                               + 8.0*(phi[cnyp1]-psi[cnyp1]) - (phi[cnyp2]-psi[cnyp2]))/(12.0*dy*dy)

							       + (f[cnzn2] - 8.0*f[cnzn1] + 8.0*f[cnzp1] - f[cnzp2])*
							     ((phi[cnzn2]-psi[cnzn2]) - 8.0*(phi[cnzn1]-psi[cnzn1])
                                                               + 8.0*(phi[cnzp1]-psi[cnzp1]) - (phi[cnzp2]-psi[cnzp2]))/(12.0*dz*dz)  
				
							   -lplf*(1.0+2.0*phi[ci])	- a_tt*f_a[ci]
                                                         )/(a_t*a_t)

		                                        +   ( 3.0*a*f_a[ci] - 3.0*a*a*phi_a[ci]*f_a[ci] 
                                                          -6.0*a*psi[ci]*f_a[ci] - a*a*f_a[ci]*psi[ci] )
						      );

               
		

		tmpphi_a[ci] = phi_a[ci] + 0.5*phiacc*da;
		tmpf_a[ci] = f_a[ci] + 0.5*facc*da;

		tmpphi[ci] = phi[ci] + 0.5*phi_a[ci]*da;
		tmpf[ci] = f[ci] + 0.5*f_a[ci]*da;

		
		

		

		++jd;
	  }


		Vvl = V(fbt);
  	  	V_fvl = V_f(fbt);
	  	a_t = sqrt((ommi*ai*ai*ai/at  + (1.0/(Mpl*Mpl))*at*at*Vvl/(3.0*c)) / 
                                                ( 1.0 - (1.0/(Mpl*Mpl))*at*at*fb_a_t*fb_a_t/(6.0*c*c*c))) ;
          	a_tt = -0.5*ommi*ai*ai*ai/(at*at) - (1.0/(Mpl*Mpl*c))*at*(fb_a_t*fb_a_t*a_t*a_t - Vvl)/3.0;
		facb = -V_fvl*c*c/(a_t*a_t) - 3.0*fb_a/a - a_tt*fb_a/(a_t*a_t);
	  
	  	fb = fb + da*fb_a;
	  	fb_a = fb_a + da*facb;
	  	a = a + da;



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


 		


		

		lplphi = (-tmpphi[cnxn2]+16.0*tmpphi[cnxn1]-30.0*tmpphi[ci]+16.0*tmpphi[cnxp1]-tmpphi[cnxp2])/(12.0*dx*dx) +
			 (-tmpphi[cnyn2]+16.0*tmpphi[cnyn1]-30.0*tmpphi[ci]+16.0*tmpphi[cnyp1]-tmpphi[cnyp2])/(12.0*dy*dy) +
			 (-tmpphi[cnzn2]+16.0*tmpphi[cnzn1]-30.0*tmpphi[ci]+16.0*tmpphi[cnzp1]-tmpphi[cnzp2])/(12.0*dz*dz)   ;

		tmppsi[ci] = (lplphi - 0.5*tmptul00[ci]/(Mpl*Mpl) - 3.0*at*a_t*tmpphi_a[ci]*a_t)/(3.0*a_t*a_t);
		tmppsi_a[ci] = (tmppsi[ci]-psi[ci])/(0.5*da);

		lplpsi = (-tmppsi[cnxn2]+16.0*tmppsi[cnxn1]-30.0*tmppsi[ci]+16.0*tmppsi[cnxp1]-tmppsi[cnxp2])/(12.0*dx*dx) +
			 (-tmppsi[cnyn2]+16.0*tmppsi[cnyn1]-30.0*tmppsi[ci]+16.0*tmppsi[cnyp1]-tmppsi[cnyp2])/(12.0*dy*dy) +
			 (-tmppsi[cnzn2]+16.0*tmppsi[cnzn1]-30.0*tmppsi[ci]+16.0*tmppsi[cnzp1]-tmppsi[cnzp2])/(12.0*dz*dz)   ;

		lplf = (-tmpf[cnxn2]+16.0*tmpf[cnxn1]-30.0*tmpf[ci]+16.0*tmpf[cnxp1]-tmpf[cnxp2])/(12.0*dx*dx) +
			 (-tmpf[cnyn2]+16.0*tmpf[cnyn1]-30.0*tmpf[ci]+16.0*tmpf[cnyp1]-tmpf[cnyp2])/(12.0*dy*dy) +
			 (-tmpf[cnzn2]+16.0*tmpf[cnzn1]-30.0*tmpf[ci]+16.0*tmpf[cnzp1]-tmpf[cnzp2])/(12.0*dz*dz)   ;

		Vvl = V(tmpf[ci]);
  	  	V_fvl = V_f(tmpf[ci]);
	
		phiacc = (0.5/(a*a))*( -2.0*tmppsi[ci] - 4.0*a*tmppsi[ci]*a_tt/(a_t*a_t) - 6.0*a*tmpphi_a[ci]- 2.0*a*tmppsi_a[ci]  + 
			               (2.0/3.0)*(lplphi-lplpsi)/(a_t*a_t) - a*a*tuldss[ci]/(Mpl*3.0*(a_t*a_t)) - tmpphi_a[ci]*a_tt/(a_t*a_t)
                                      );

		facc = (1.0/(a*a*(-1.0+2.0*tmppsi[ci]))) *(  ( a*a*V_fvl 

                                                           + (tmpf[cnxn2] - 8.0*tmpf[cnxn1] + 8.0*tmpf[cnxp1] - tmpf[cnxp2])*
							     ((tmpphi[cnxn2]-tmppsi[cnxn2]) - 8.0*(tmpphi[cnxn1]-tmppsi[cnxn1])
                                                               + 8.0*(tmpphi[cnxp1]-tmppsi[cnxp1]) - (tmpphi[cnxp2]-tmppsi[cnxp2]))/(dx*dx)

							       + (tmpf[cnyn2] - 8.0*tmpf[cnyn1] + 8.0*tmpf[cnyp1] - tmpf[cnyp2])*
							     ((tmpphi[cnyn2]-tmppsi[cnyn2]) - 8.0*(tmpphi[cnyn1]-tmppsi[cnyn1])
                                                               + 8.0*(tmpphi[cnyp1]-tmppsi[cnyp1]) - (tmpphi[cnyp2]-tmppsi[cnyp2]))/(dy*dy)

							       + (tmpf[cnzn2] - 8.0*tmpf[cnzn1] + 8.0*tmpf[cnzp1] - tmpf[cnzp2])*
							     ((tmpphi[cnzn2]-tmppsi[cnzn2]) - 8.0*(tmpphi[cnzn1]-tmppsi[cnzn1])
                                                               + 8.0*(tmpphi[cnzp1]-tmppsi[cnzp1]) - (tmpphi[cnzp2]-tmppsi[cnzp2]))/(dz*dz)  
				
							   -lplf*(1.0+2.0*tmpphi[ci])	- a_tt*tmpf_a[ci]
                                                         )/(a_t*a_t)

		                                        +   ( 3.0*a*tmpf_a[ci] - 3.0*a*a*tmpphi_a[ci]*tmpf_a[ci] 
                                                          -6.0*a*tmppsi[ci]*tmpf_a[ci] - a*a*tmpf_a[ci]*tmppsi[ci] )
						      );

                phi_a[ci] = phi_a[ci] + phiacc*da;
		f_a[ci] = f_a[ci] + facc*da;

		phi[ci] = phi[ci] + tmpphi_a[ci]*da;
		f[ci] = f[ci] + tmpf_a[ci]*da;

		psi[ci] = psi[ci] + tmppsi_a[ci]*da;
		psi_a[ci] = (psi[ci]-tmppsi[ci])/(0.5*da);
	

		
		

		

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


