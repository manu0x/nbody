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

		nic[i][12] = (i+(n*n)+n)%N;
		nic[i][13] = (i+n+1)%N;
		nic[i][14] = (i+(n*n)+1)%N;
		nic[i][15] = (N+i+(n*n)+n+1)%N;
		


         }




}


void initialise()
{
      int ix,iy,iz,Vvl,V_fvl;

      double cpmc = (0.14/(0.68*0.68));
      int px,py,pz,ci,pgi;
      double gamma, v, gradmagf;
      a0 = 1000.0;
      ai = 1.0;
      fb = 1.0;
      fb_a = 0.0;
      omdmb= (cpmc)*pow((a0/ai),3.0)/(cpmc*a0*a0*a0/(ai*ai*ai) + (1.0-cpmc));

	dx = 0.001; dy =0.001; dz = 0.001;
        L = dx*((double) n); 
      
	for(ci = 0;ci <N; ++ci)
	{
		

       				
				phi[ci] = (double) rand()/((double) RAND_MAX  );
      				
				
				psi[ci] = (double) rand()/((double) RAND_MAX  );
      				
		
				f[ci] = fb;
      				

				phi_a[ci] = 0.0;
      				
				
				psi_a[ci] = 0.0;
      				
		
				f_a[ci] = fb_a;

				p[ci][0] =  ((double) rand()/((double) RAND_MAX  ))*L;
				px = (int)(p[ci][0]/dx);
				p[ci][1] =  ((double) rand()/((double) RAND_MAX  ))*L;
				py = (int)(p[ci][1]/dy);
				p[ci][2] =  ((double) rand()/((double) RAND_MAX  ))*L;
				pz = (int)(p[ci][3]/dx);

				pgi = px*n*n+py*n+pz;

				p[ci][4] =  0.0;
				p[ci][5] =  0.0;
				p[ci][6] =  0.0;

				v = sqrt(p[ci][3]*p[ci][3] + p[ci][4]*p[ci][4]
                                         + p[ci][5]*p[ci][5]);
				gamma = 1.0/sqrt(1.0-a*a*v*v);


//////////////////////////////////particle tul calculation begins/////////////////////////////////////////////////////////////////////////////////

				
      				tul00[pgi]+= (1.0/(8.0))* m[ci]*gamma*(1.0 + 3.0*phi[pgi] - psi[pgi]  
                                                            -gamma*gamma*(v*v*a*a*phi[pgi] + psi[pgi])  );
		tul00[nic[pgi][3]]+=(1.0/(8.0))*  m[ci]*gamma*(1.0 + 3.0*phi[nic[pgi][3]] - psi[nic[pgi][3]]  
                                                            -gamma*gamma*(v*v*a*a*phi[nic[pgi][3]] + psi[nic[pgi][3]])  );
		tul00[nic[pgi][4]]+=(1.0/(8.0))*  m[ci]*gamma*(1.0 + 3.0*phi[nic[pgi][4]] - psi[nic[pgi][4]]  
                                                            -gamma*gamma*(v*v*a*a*phi[nic[pgi][4]] + psi[nic[pgi][4]])  );
		tul00[nic[pgi][5]]+=  (1.0/(8.0))*m[ci]*gamma*(1.0 + 3.0*phi[nic[pgi][5]] - psi[nic[pgi][5]]  
                                                            -gamma*gamma*(v*v*a*a*phi[nic[pgi][5]] + psi[nic[pgi][5]])  );
		tul00[nic[pgi][12]]+= 
                                      (1.0/(8.0))* m[ci]*gamma*(1.0 + 3.0*phi[nic[pgi][12]] - psi[nic[pgi][12]]  
                                                            -gamma*gamma*(v*v*a*a*phi[nic[pgi][12]] + psi[nic[pgi][12]])  );
				tul00[nic[pgi][14]]+= 
					(1.0/(8.0))*m[ci]*gamma*(1.0 + 3.0*phi[nic[pgi][14]] - psi[nic[pgi][14]]  
                                                            -gamma*gamma*(v*v*a*a*phi[nic[pgi][14]] + psi[nic[pgi][14]])  );
				tul00[nic[pgi][13]]+=  
					(1.0/(8.0))*m[ci]*gamma*(1.0 + 3.0*phi[nic[pgi][13]] - psi[nic[pgi][13]]  
                                                            -gamma*gamma*(v*v*a*a*phi[nic[pgi][13]] + psi[nic[pgi][13]])  );
				tul00[nic[pgi][15]]+=  
				(1.0/(8.0))*m[ci]*gamma*(1.0 + 3.0*phi[nic[pgi][15]] - psi[nic[pgi][15]]  
                                                    -gamma*gamma*(v*v*a*a*phi[nic[pgi][15]] + psi[nic[pgi][15]])  );
//////////////////////////////////////diagonal sum of spatial tu//////////////////////////////////////////////////////////////////////////////////
				tuldss[pgi]+= (1.0/(24.0))*m[ci]*gamma*v*v*
								(1.0 + 3.0*phi[pgi] - psi[pgi]  
                                                            -gamma*v*v*gamma*v*v*(v*v*a*a*phi[pgi] + psi[pgi])  );
				tuldss[nic[pgi][3]]+= 
						         (1.0/(24.0))*m[ci]*gamma*v*v*
									(1.0 + 3.0*phi[nic[pgi][3]] - psi[nic[pgi][3]]  
                                                           -gamma*v*v*gamma*v*v*(v*v*a*a*phi[nic[pgi][3]] + psi[nic[pgi][3]])  );
				tuldss[nic[pgi][4]]+=  
							 (1.0/(24.0))*m[ci]*gamma*v*v*
									(1.0 + 3.0*phi[nic[pgi][4]] - psi[nic[pgi][4]]  
                                                             -gamma*v*v*gamma*v*v*(v*v*a*a*phi[nic[pgi][4]] + psi[nic[pgi][4]])  );
				tuldss[nic[pgi][5]]+=  
							(1.0/(24.0))*m[ci]*gamma*v*v*
									(1.0 + 3.0*phi[nic[pgi][5]] - psi[nic[pgi][5]]  
                                                            -gamma*v*v*gamma*v*v*(v*v*a*a*phi[nic[pgi][5]] + psi[nic[pgi][5]])  );
				tuldss[nic[pgi][12]]+= 
                                                 (1.0/(24.0))*m[ci]*gamma*v*v*
								(1.0 + 3.0*phi[nic[pgi][12]] - psi[nic[pgi][12]]  
                                                       -gamma*v*v*gamma*v*v*(v*v*a*a*phi[nic[pgi][12]] + psi[nic[pgi][12]])  );
				tuldss[nic[pgi][14]]+= 
						 (1.0/(24.0))*m[ci]*gamma*v*v*
									(1.0 + 3.0*phi[nic[pgi][14]] - psi[nic[pgi][14]]  
                                                       -gamma*v*v*gamma*v*v*(v*v*a*a*phi[nic[pgi][14]] + psi[nic[pgi][14]])  );
				tuldss[nic[pgi][13]]+=  
						 (1.0/(24.0))*m[ci]*gamma*v*v*
								(1.0 + 3.0*phi[nic[pgi][13]] - psi[nic[pgi][13]]  
                                                       -gamma*v*v*gamma*v*v*(v*v*a*a*phi[nic[pgi][13]] + psi[nic[pgi][13]])  );
				tuldss[nic[pgi][15]]+=  
					(1.0/(18.0))*m[ci]*gamma*v*v*
						(1.0 + 3.0*phi[nic[pgi][15]] - psi[nic[pgi][15]]  
                                              -gamma*v*v*gamma*v*v*(v*v*a*a*phi[nic[pgi][15]] + psi[nic[pgi][15]])  );
///////////////////////////////////particle tul calculation end/////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////Quint field tul cal begins with tul00/////////////////////////////////////////////////////////////////////
				

				Vvl = V(f[ci]);
				gradmagf =  ((f[nic[ci][6]] - 8.0*f[nic[ci][0]] + 8.0*f[nic[ci][3]] - f[nic[ci][9]])
							   /(12.0*dx))*((f[nic[ci][6]] - 8.0*f[nic[ci][0]] + 8.0*f[nic[ci][3]] - f[nic[ci][9]])
							    /(12.0*dx))
					   + ((f[nic[ci][7]] - 8.0*f[nic[ci][1]] + 8.0*f[nic[ci][4]] - f[nic[ci][10]])
				            /(12.0*dy))*((f[nic[ci][7]] - 8.0*f[nic[ci][1]] + 8.0*f[nic[ci][4]] - f[nic[ci][10]])
							    /(12.0*dy))
					  + ((f[nic[ci][8]] - 8.0*f[nic[ci][2]] + 8.0*f[nic[ci][5]] - f[nic[ci][11]])
							    /(12.0*dz))*((f[nic[ci][8]] - 8.0*f[nic[ci][2]] + 8.0*f[nic[ci][5]] - f[nic[ci][11]])
							    /(12.0*dz));

								   

				tul00[ci]  = Vvl + 0.5*f_a[ci]*f_a[ci]*a_t*a_t*(1.0-2.0*psi[ci])
							  + 0.5*gradmagf*(1.0+2.0*phi[ci])/(a*a);
				tuldss[ci] = Vvl - 0.5*f_a[ci]*f_a[ci]*a_t*a_t*(1.0-2.0*psi[ci])
							  + gradmagf*(1.0+2.0*phi[ci])/(6.0*a*a);	
		

				
				
	
      				

			
	}
}


int evolve(double aini, double astp)
{
    


    double ommi = omdmb;
    double facb;
    double w;

    int fail = 1,i,j;

    double nd = (double) n, jd;  ///Watch out for local vs global for parallelization
    double phiacc,facc,pacc[3],Vvl,V_fvl,v,gamma,phiavg,psiavg,phi_a_avg,psi_a_avg,grdavgpsi,grdavgphi;
    int ix,iy,iz,ci,px,py,pz,pgi;
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
	  for(ci=0;ci<n;++ci)
	  {     jd = ci;
		ix = (int) (floor(jd/(nd*nd)));
		iy = (int) (floor(fmod(jd,(nd*nd))/nd));
		iz = (int) (jd-((double) ix)*nd*nd-((double) iy)*nd);
		
		
		lplphi = (-phi[nic[ci][6]]+16.0*phi[nic[ci][0]]-30.0*phi[ci]+16.0*phi[nic[ci][3]]-phi[nic[ci][9]])/(12.0*dx*dx) +
			 (-phi[nic[ci][7]]+16.0*phi[nic[ci][1]]-30.0*phi[ci]+16.0*phi[nic[ci][4]]-phi[nic[ci][10]])/(12.0*dy*dy) +
			 (-phi[nic[ci][8]]+16.0*phi[nic[ci][2]]-30.0*phi[ci]+16.0*phi[nic[ci][5]]-phi[nic[ci][11]])/(12.0*dz*dz)   ;

		lplpsi = (-psi[nic[ci][6]]+16.0*psi[nic[ci][0]]-30.0*psi[ci]+16.0*psi[nic[ci][3]]-psi[nic[ci][9]])/(12.0*dx*dx) +
			 (-psi[nic[ci][7]]+16.0*psi[nic[ci][1]]-30.0*psi[ci]+16.0*psi[nic[ci][4]]-psi[nic[ci][10]])/(12.0*dy*dy) +
			 (-psi[nic[ci][8]]+16.0*psi[nic[ci][2]]-30.0*psi[ci]+16.0*psi[nic[ci][5]]-psi[nic[ci][11]])/(12.0*dz*dz)   ;

		lplf = (-f[nic[ci][6]]+16.0*f[nic[ci][0]]-30.0*f[ci]+16.0*f[nic[ci][3]]-f[nic[ci][9]])/(12.0*dx*dx) +
			 (-f[nic[ci][7]]+16.0*f[nic[ci][1]]-30.0*f[ci]+16.0*f[nic[ci][4]]-f[nic[ci][10]])/(12.0*dy*dy) +
			 (-f[nic[ci][8]]+16.0*f[nic[ci][2]]-30.0*f[ci]+16.0*f[nic[ci][5]]-f[nic[ci][11]])/(12.0*dz*dz)   ;

		Vvl = V(f[ci]);
  	  	V_fvl = V_f(f[ci]);
	
		phiacc = (0.5/(a*a))*( -2.0*psi[ci] - 4.0*a*psi[ci]*a_tt/(a_t*a_t) - 6.0*a*phi_a[ci]- 2.0*a*psi_a[ci]  + 
			               (2.0/3.0)*(lplphi-lplpsi)/(a_t*a_t) - a*a*tuldss[ci]/(Mpl*3.0*(a_t*a_t)) - phi_a[ci]*a_tt/(a_t*a_t)
                                      );

		facc = (1.0/(a*a*(-1.0+2.0*psi[ci]))) *(  ( a*a*V_fvl 

                                                        + (f[nic[ci][6]] - 8.0*f[nic[ci][0]] + 8.0*f[nic[ci][3]] - f[nic[ci][9]])*
							 ((phi[nic[ci][6]]-psi[nic[ci][6]]) - 8.0*(phi[nic[ci][0]]-psi[nic[ci][0]])
                                                     + 8.0*(phi[nic[ci][3]]-psi[nic[ci][3]]) - (phi[nic[ci][9]]-psi[nic[ci][9]]))/(144.0*dx*dx)

							       + (f[nic[ci][7]] - 8.0*f[nic[ci][1]] + 8.0*f[nic[ci][4]] - f[nic[ci][10]])*
							     ((phi[nic[ci][7]]-psi[nic[ci][7]]) - 8.0*(phi[nic[ci][1]]-psi[nic[ci][1]])
                                                    + 8.0*(phi[nic[ci][4]]-psi[nic[ci][4]]) - (phi[nic[ci][10]]-psi[nic[ci][10]]))/(144.0*dy*dy)

							       + (f[nic[ci][8]] - 8.0*f[nic[ci][2]] + 8.0*f[nic[ci][5]] - f[nic[ci][11]])*
							     ((phi[nic[ci][8]]-psi[nic[ci][8]]) - 8.0*(phi[nic[ci][2]]-psi[nic[ci][2]])
                                                + 8.0*(phi[nic[ci][5]]-psi[nic[ci][5]]) - (phi[nic[ci][11]]-psi[nic[ci][11]]))/(144.0*dz*dz)  
				
							   -lplf*(1.0+2.0*phi[ci])	- a_tt*f_a[ci]
                                                         )/(a_t*a_t)

		                                        +   ( 3.0*a*f_a[ci] - 3.0*a*a*phi_a[ci]*f_a[ci] 
                                                          -6.0*a*psi[ci]*f_a[ci] - a*a*f_a[ci]*psi[ci] )
						      );

///////////////////////////////////////////////particle accelerations////////////////////////////////////////////////////////////////////////////
		v = sqrt(p[ci][3]*p[ci][3] + p[ci][4]*p[ci][4]
                                         + p[ci][5]*p[ci][5]);
				gamma = 1.0/sqrt(1.0-a*a*v*v);
		
			px = (int)(p[ci][0]/dx);
			py = (int)(p[ci][1]/dy);
			pz = (int)(p[ci][3]/dx);

			pgi = px*n*n+py*n+pz;

		psiavg = (1.0/8.0)*(psi[pgi] + psi[nic[pgi][3]]+ psi[nic[pgi][4]]+ psi[nic[pgi][5]]+ 
                         psi[nic[pgi][12]]+psi[nic[pgi][13]]+ psi[nic[pgi][14]]+ psi[nic[pgi][15]]);
               
		phiavg = (1.0/8.0)*(phi[pgi] + phi[nic[pgi][3]]+ phi[nic[pgi][4]]+ phi[nic[pgi][5]]+ 
                         phi[nic[pgi][12]]+phi[nic[pgi][13]]+ phi[nic[pgi][14]]+ phi[nic[pgi][15]]);

		psi_a_avg = (1.0/8.0)*(psi_a[pgi] + psi_a[nic[pgi][3]]+ psi_a[nic[pgi][4]]+ psi_a[nic[pgi][5]]+ 
                         psi_a[nic[pgi][12]]+psi_a[nic[pgi][13]]+ psi_a[nic[pgi][14]]+ psi_a[nic[pgi][15]]);
               
		phi_a_avg = (1.0/8.0)*(phi_a[pgi] + phi_a[nic[pgi][3]]+ phi_a[nic[pgi][4]]+ phi_a[nic[pgi][5]]+ 
                         phi_a[nic[pgi][12]]+phi_a[nic[pgi][13]]+ phi_a[nic[pgi][14]]+ phi_a[nic[pgi][15]]);

		grdavgpsi = psi[nic[pgi][3]] - psi[pgi] + psi[nic[pgi][12]] - psi[nic[pgi][4]] 
               
		pacc = p[ci][3]*v*v*(-2.0*a*a_t*phiavg - 2.0*a*a_t*psiavg - a*a*phi_a_avg + a*a_t) 
			+ p[ci][3]*(2.0*(   ))
///////////////////////////////////////////////////////////particle acceleration calculation ends////////////////////////////////////////////////
		tmpphi_a[ci] = phi_a[ci] + 0.5*phiacc*da;
		tmpf_a[ci] = f_a[ci] + 0.5*facc*da;

		tmpphi[ci] = phi[ci] + 0.5*phi_a[ci]*da;
		tmpf[ci] = f[ci] + 0.5*f_a[ci]*da;

		
		

		

		
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
		nic[ci][3] = (n + (ix+1))%n;
		nic[ci][9] = (n + (ix+2))%n;
		nic[ci][0] = (n + (ix-1))%n;
		nic[ci][6] = (n + (ix-2))%n;
		nic[ci][4] = (n + (iy+1))%n;
		nic[ci][10] = (n + (iy+2))%n;
		nic[ci][1] = (n + (iy-1))%n;
		nic[ci][7] = (n + (iy-2))%n;
		nic[ci][5] = (n + (iz+1))%n;
		nic[ci][11] = (n + (iz+2))%n;
		nic[ci][2] = (n + (iz-1))%n;
		nic[ci][8] = (n + (iz-2))%n;


 		


		

		lplphi = (-tmpphi[nic[ci][6]]+16.0*tmpphi[nic[ci][0]]-30.0*tmpphi[ci]+16.0*tmpphi[nic[ci][3]]-tmpphi[nic[ci][9]])/(12.0*dx*dx) +
			 (-tmpphi[nic[ci][7]]+16.0*tmpphi[nic[ci][1]]-30.0*tmpphi[ci]+16.0*tmpphi[nic[ci][4]]-tmpphi[nic[ci][10]])/(12.0*dy*dy) +
			 (-tmpphi[nic[ci][8]]+16.0*tmpphi[nic[ci][2]]-30.0*tmpphi[ci]+16.0*tmpphi[nic[ci][5]]-tmpphi[nic[ci][11]])/(12.0*dz*dz)   ;

		tmppsi[ci] = (lplphi - 0.5*tmptul00[ci]/(Mpl*Mpl) - 3.0*at*a_t*tmpphi_a[ci]*a_t)/(3.0*a_t*a_t);
		tmppsi_a[ci] = (tmppsi[ci]-psi[ci])/(0.5*da);

		lplpsi = (-tmppsi[nic[ci][6]]+16.0*tmppsi[nic[ci][0]]-30.0*tmppsi[ci]+16.0*tmppsi[nic[ci][3]]-tmppsi[nic[ci][9]])/(12.0*dx*dx) +
			 (-tmppsi[nic[ci][7]]+16.0*tmppsi[nic[ci][1]]-30.0*tmppsi[ci]+16.0*tmppsi[nic[ci][4]]-tmppsi[nic[ci][10]])/(12.0*dy*dy) +
			 (-tmppsi[nic[ci][8]]+16.0*tmppsi[nic[ci][2]]-30.0*tmppsi[ci]+16.0*tmppsi[nic[ci][5]]-tmppsi[nic[ci][11]])/(12.0*dz*dz)   ;

		lplf = (-tmpf[nic[ci][6]]+16.0*tmpf[nic[ci][0]]-30.0*tmpf[ci]+16.0*tmpf[nic[ci][3]]-tmpf[nic[ci][9]])/(12.0*dx*dx) +
			 (-tmpf[nic[ci][7]]+16.0*tmpf[nic[ci][1]]-30.0*tmpf[ci]+16.0*tmpf[nic[ci][4]]-tmpf[nic[ci][10]])/(12.0*dy*dy) +
			 (-tmpf[nic[ci][8]]+16.0*tmpf[nic[ci][2]]-30.0*tmpf[ci]+16.0*tmpf[nic[ci][5]]-tmpf[nic[ci][11]])/(12.0*dz*dz)   ;

		Vvl = V(tmpf[ci]);
  	  	V_fvl = V_f(tmpf[ci]);
	
		phiacc = (0.5/(a*a))*( -2.0*tmppsi[ci] - 4.0*a*tmppsi[ci]*a_tt/(a_t*a_t) - 6.0*a*tmpphi_a[ci]- 2.0*a*tmppsi_a[ci]  + 
			               (2.0/3.0)*(lplphi-lplpsi)/(a_t*a_t) - a*a*tuldss[ci]/(Mpl*3.0*(a_t*a_t)) - tmpphi_a[ci]*a_tt/(a_t*a_t)
                                      );

		facc = (1.0/(a*a*(-1.0+2.0*tmppsi[ci]))) *(  ( a*a*V_fvl 

                                                        + (tmpf[nic[ci][6]] - 8.0*tmpf[nic[ci][0]] + 8.0*tmpf[nic[ci][3]] - tmpf[nic[ci][9]])*
							  ((tmpphi[nic[ci][6]]-tmppsi[nic[ci][6]]) - 8.0*(tmpphi[nic[ci][0]]-tmppsi[nic[ci][0]])
                                                   + 8.0*(tmpphi[nic[ci][3]]-tmppsi[nic[ci][3]]) - (tmpphi[nic[ci][9]]-tmppsi[nic[ci][9]]))/(dx*dx)

							       + (tmpf[nic[ci][7]] - 8.0*tmpf[nic[ci][1]] + 8.0*tmpf[nic[ci][4]] - tmpf[nic[ci][10]])*
							     ((tmpphi[nic[ci][7]]-tmppsi[nic[ci][7]]) - 8.0*(tmpphi[nic[ci][1]]-tmppsi[nic[ci][1]])
                                                          + 8.0*(tmpphi[nic[ci][4]]-tmppsi[nic[ci][4]]) - (tmpphi[nic[ci][10]]-tmppsi[nic[ci][10]]))/(dy*dy)

							       + (tmpf[nic[ci][8]] - 8.0*tmpf[nic[ci][2]] + 8.0*tmpf[nic[ci][5]] - tmpf[nic[ci][11]])*
							     ((tmpphi[nic[ci][8]]-tmppsi[nic[ci][8]]) - 8.0*(tmpphi[nic[ci][2]]-tmppsi[nic[ci][2]])
                                                        + 8.0*(tmpphi[nic[ci][5]]-tmppsi[nic[ci][5]]) - (tmpphi[nic[ci][11]]-tmppsi[nic[ci][11]]))/(dz*dz)  
				
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


