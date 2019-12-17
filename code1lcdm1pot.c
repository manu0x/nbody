#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h> 
#include <omp.h>
#include <fenv.h>
#include <time.h>
#include "mt19937ar.c"

#define  n 32

double G   = 1.0;
double c   = 1.0;
double Mpl ;
double H0  = 22.04*(1e-5);
double L[3];
int tN;


double *phi, *phi, *f,*phi_a, *phi_a, *f_a,*tul00,*tuldss;
double phi_s[3][n*n*n],phi_s[3][n*n*n],f_s[3][n*n*n],LAPphi[n*n*n],LAPphi[n*n*n],LAPf[n*n*n],usty[n*n*n],psty[n*n*n];
double *tmpphi, *tmpphi, *tmpf,*tmpphi_a, *tmpphi_a, *tmpf_a, *ini_vel0,*ini_vel1,*ini_vel2,m=1.0;
double dx[3];
struct particle
	{	
		double x[3];
		double v[3];
		
		
		int 	cubeind[8];	

	};



struct particle p[n*n*n],tmpp[n*n*n];
double grid[n*n*n][3];


 
  
  

fftw_complex *ini_del;
fftw_complex *F_ini_del;
fftw_complex *ini_del;
fftw_complex *F_ini_phi;
fftw_complex *ini_phi;
fftw_complex *F_ini_v0;
fftw_complex *ini_v0;
fftw_complex *F_ini_v1;
fftw_complex *ini_v1;
fftw_complex *F_ini_v2;
fftw_complex *ini_v2;


fftw_plan ini_del_plan;
fftw_plan ini_phi_plan;
fftw_plan ini_v0_plan;
fftw_plan ini_v1_plan;
fftw_plan ini_v2_plan;

int gridind[n*n*n][3];

int nic[n*n*n][16];

double  omdmb, omdeb, a, ak, a_t, a_tt, Vamp, ai, a0, da, ak, fbt, fb_at,ommi,omdei;
int jprint;
double H0, Hi;

FILE *fpback;


void background();




void initialise();
double ini_power_spec(double);
void ini_rand_field();
double mesh2particle(struct particle *,int,double *);
void particle2mesh(struct particle * ,int ,double *,double );
int evolve(double ,double );

void main()
{       Mpl = 1.0/sqrt(8.0*3.142*G) ;
        da = 0.01;
        jprint = (int) (1.0/da);
	printf("jprint %d\n",jprint);
	tN=n*n*n;
         
	//feenableexcept(FE_DIVBYZERO | FE_ItNVALID | FE_OVERFLOW);

	

	fpback  = fopen("back.txt","w");

        int i;

       // i = fftw_init_threads();
	//	fftw_plan_with_nthreads(omp_get_max_threads());

	phi = (double *) malloc(n*n*n*sizeof(double)); 
        phi_a = (double *) malloc(n*n*n*sizeof(double)); 
	phi = (double *) malloc(n*n*n*sizeof(double)); 
        phi_a = (double *) malloc(n*n*n*sizeof(double)); 
	//f = (double *) malloc(n*n*n*sizeof(double)); 
        //f_a = (double *) malloc(n*n*n*sizeof(double)); 
	tul00 = (double *) malloc(n*n*n*sizeof(double)); 
        tuldss = (double *) malloc(n*n*n*sizeof(double));

	tmpphi = (double *) malloc(n*n*n*sizeof(double)); 
        tmpphi_a = (double *) malloc(n*n*n*sizeof(double)); 
	tmpphi = (double *) malloc(n*n*n*sizeof(double)); 
        tmpphi_a = (double *) malloc(n*n*n*sizeof(double)); 
	//tmpf = (double *) malloc(n*n*n*sizeof(double)); 
        //tmpf_a = (double *) malloc(n*n*n*sizeof(double)); 
 

	ini_vel0=(double *) malloc(n*n*n*sizeof(double));
	ini_vel1=(double *) malloc(n*n*n*sizeof(double));
	ini_vel2=(double *) malloc(n*n*n*sizeof(double));

        F_ini_del = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n*n*n);
	ini_del = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n*n*n);
	F_ini_phi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n*n*n);
	ini_phi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n*n*n);
	F_ini_v0 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n*n*n);
	ini_v0 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n*n*n);
	F_ini_v1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n*n*n);
	ini_v1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n*n*n);
	F_ini_v2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n*n*n);
	ini_v2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n*n*n);
	//m = (double *) malloc(n*n*n*sizeof(double)); 
	
	
      
	
	background();
	initialise();
	

      // i = evolve(ai,100.0);
      




}


double mesh2particle(struct particle *pp,int p_id,double *meshf)
{

	int i,j,k;
	
	double rv=0.0,del;
	
	
	
	
	for(i=0;i<8;++i)
	{
		k = pp[p_id].cubeind[i];
		del = 1.0;
		for(j=0;j<3;++j)
		del*=  (fabs(pp[p_id].x[j]-grid[k][j])/dx[j])*(fabs(pp[p_id].x[j]-grid[k][j])/dx[j]);
			
		//printf("del %lf\n",del);		
		rv+= del*meshf[k];
		



	}
	

	return(rv);

}




void cal_spectrum(double *spcmesh)
{	int i,j;

	fftw_complex *dens_cntrst; fftw_complex *Fdens_cntrst;
	fftw_plan spec_plan;

	dens_cntrst = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n*n*n);
	Fdens_cntrst = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n*n*n);

	for(i=0;i<tn;++i)
	{
		dens_cntrs[i][0] = spcmesh[i];
		dens_cntrs[i][1] = 0.0;
	}


	spec_plan = fftw_plan_dft_3d(n,n,n, dens_cntrs, Fdens_cntrst, FFTW_FORWARD, FFTW_ESTIMATE);


}



double ini_power_spec(double kamp)
{


	return(0.011);


}

void ini_rand_field()
{	init_genrand(time(0));
	int i,j,k,ief,jef,kef,cnt,rcnt,rk,maxcnt=0; 
	double ksqr,muk,sigk;
	double a1,a2,b1,b2,a,b;


	init_genrand(time(0));
	
	for(i=0;i<n;++i)
	{	if(i<(n/2))
		ief = i;
		else
		ief = i-n+1;

		for(j=0;j<(n);++j)
		{
			if(i<(n/2))
			jef = j;
			else
			jef = j-n+1;
			for(k=0;k<(n/2);++k)
			{   cnt = i*n*n + j*n + k;
				if(maxcnt<cnt)
			    	 maxcnt = cnt;
			   ksqr = 2.0*M_PI*2.0*M_PI*
				(((double) ief)*((double) ief)/(dx[0]*dx[0]) 
				+ ((double) jef)*((double) jef)/(dx[1]*dx[1])+ ((double) k)*((double) k)/(dx[2]*dx[2]) ) 
					/(((double) n)*((double) n));
			  sigk  = ini_power_spec(sqrt(ksqr));
			  muk = sigk/sqrt(2.0);
		 	a1 = genrand_res53();
 			a2 = genrand_res53(); 
			b1 = genrand_res53();
 			b2 = genrand_res53();
			a = (muk*(sqrt(-2.0*log(a1))*cos(2.0*M_PI*a2)));
			b = (muk*(sqrt(-2.0*log(b1))*cos(2.0*M_PI*b2)));
				
			F_ini_del[cnt][0] = a;	F_ini_del[cnt][1] = b;

			F_ini_phi[cnt][0] = -1.5*omdmb*Hi*Hi*ai*ai*F_ini_del[cnt][0]/ksqr;	
			F_ini_phi[cnt][1] = -1.5*omdmb*Hi*Hi*ai*ai*F_ini_del[cnt][1]/ksqr;

			

			F_ini_v2[cnt][0] =  -k*F_ini_phi[cnt][1]/(dx[2]*n);
			F_ini_v2[cnt][1] =  k*F_ini_phi[cnt][0]/(dx[2]*n);


  		 	// if(k!=0)
			  {

			  	rk = (n-k)-1;
				rcnt = i*n*n + j*n + rk; 
				if(maxcnt<rcnt)
			    	 maxcnt = rcnt;

				F_ini_del[rcnt][0] = F_ini_del[rcnt][0];	F_ini_del[rcnt][1] = -F_ini_del[cnt][1];
				F_ini_phi[rcnt][0] = F_ini_phi[rcnt][0];	F_ini_phi[rcnt][1] = -F_ini_phi[cnt][1];
				
				
				F_ini_v2[rcnt][0] = F_ini_v2[rcnt][0];	F_ini_v2[rcnt][1] = -F_ini_v2[cnt][1];
	
			  }


			}


		}


	}

	//printf("maxx %d\n",maxcnt);

	for(i=0;i<(n);++i)
	{
		for(k=0;k<(n);++k)
		{
			for(j=0;j<(n/2);++j)
			{  
				cnt = i*n*n + j*n + k;
				rk = (n-j)-1;
				rcnt = i*n*n + rk*n + k;

				F_ini_v1[cnt][0] =  -j*F_ini_phi[cnt][1]/(dx[1]*n);
				F_ini_v1[cnt][1] =  j*F_ini_phi[cnt][0]/(dx[1]*n);

				F_ini_v1[rcnt][0] = F_ini_v2[rcnt][0];	F_ini_v2[rcnt][1] = -F_ini_v2[cnt][1];
			}

		

		}
	}



	for(j=0;j<(n);++j)
	{
		for(k=0;k<(n);++k)
		{
			for(i=0;i<(n/2);++i)
			{  
				cnt = i*n*n + j*n + k;
				rk = (n-i)-1;
				rcnt = rk*n*n + j*n + k;

				F_ini_v0[cnt][0] =  -i*F_ini_phi[cnt][1]/(dx[0]*n);
				F_ini_v0[cnt][1] =  i*F_ini_phi[cnt][0]/(dx[0]*n);

				F_ini_v0[rcnt][0] = F_ini_v0[rcnt][0];	F_ini_v0[rcnt][1] = -F_ini_v0[cnt][1];
			}

		

		}
	}



	ini_del_plan = fftw_plan_dft_3d(n,n,n, F_ini_del, ini_del, FFTW_BACKWARD, FFTW_ESTIMATE);
	ini_v0_plan = fftw_plan_dft_3d(n,n,n, F_ini_v0, ini_v0, FFTW_BACKWARD, FFTW_ESTIMATE);
	ini_v1_plan = fftw_plan_dft_3d(n,n,n, F_ini_v1, ini_v1, FFTW_BACKWARD, FFTW_ESTIMATE);
	ini_v2_plan = fftw_plan_dft_3d(n,n,n, F_ini_v2, ini_v2, FFTW_BACKWARD, FFTW_ESTIMATE);
	ini_phi_plan = fftw_plan_dft_3d(n,n,n, F_ini_phi, ini_phi, FFTW_BACKWARD, FFTW_ESTIMATE);
	

	fftw_execute(ini_del_plan);
	fftw_execute(ini_phi_plan);
	fftw_execute(ini_v0_plan);
	fftw_execute(ini_v1_plan);
	fftw_execute(ini_v2_plan);

	
	for(cnt=0;cnt<tN;++cnt)
	{

		ini_del[cnt][0] = ini_del[cnt][0]/n; ini_del[cnt][1] = ini_del[cnt][1]/n; 
		ini_phi[cnt][0] = ini_phi[cnt][0]/n; ini_phi[cnt][1] = ini_phi[cnt][1]/n;
		ini_v0[cnt][0] = ini_v0[cnt][0]/n;   ini_v0[cnt][1] = ini_v0[cnt][1]/n;
		ini_v1[cnt][0] = ini_v1[cnt][0]/n;   ini_v1[cnt][1] = ini_v1[cnt][1]/n;
		ini_v2[cnt][0] = ini_v2[cnt][0]/n;   ini_v2[cnt][1] = ini_v2[cnt][1]/n;

		ini_vel0[cnt] = ini_v0[cnt][0];
		ini_vel1[cnt] = ini_v1[cnt][0];
		ini_vel2[cnt] = ini_v2[cnt][0];  

	}

	 fftw_free(F_ini_phi);
	 fftw_free(F_ini_del);
	 fftw_free(F_ini_v0);
	 fftw_free(F_ini_v1);
	 fftw_free(F_ini_v2);


	fftw_destroy_plan(ini_del_plan);
	fftw_destroy_plan(ini_phi_plan);
	fftw_destroy_plan(ini_v0_plan);
	fftw_destroy_plan(ini_v1_plan);
	fftw_destroy_plan(ini_v2_plan);

	printf("Generated initial Gaussian Random field\n");
	
}



void ini_displace_particle()
{	double iniv0,iniv1,iniv2,ds=0.001;
	int i,ci;

	 for(ci=0;ci<tN;++ci)
	  {
			

	
		iniv0 = mesh2particle(p,ci,ini_vel0);
		iniv1 = mesh2particle(p,ci,ini_vel1);
		iniv2 = mesh2particle(p,ci,ini_vel2);


		p[ci].v[0]  = iniv0/a_t;
		p[ci].v[1]  = iniv1/a_t;
		p[ci].v[2]  = iniv2/a_t;
		
		for(i=0;i<3;++i)
		{	p[ci].x[i] = p[ci].x[i] + ds*p[ci].v[i];
			if((p[ci].x[i]>L[i])||(p[ci].x[i]<0.0))
				p[ci].x[i] = fmod(p[ci].x[i]+L[i],L[i]);
		}
	 }


}

void particle2mesh(struct particle * pp,int p_id,double *meshphi,double ap)
{

	int i,j,k;
	int anchor[3];
	double rvphi=0.0,del[8];
	double gamma,vmgsqr;
	
	vmgsqr=a_t*a_t*(pp[p_id].v[0]*pp[p_id].v[0]+pp[p_id].v[1]*pp[p_id].v[1]+pp[p_id].v[2]*pp[p_id].v[2]);
	gamma = 1.0/sqrt(1.0-ap*ap*vmgsqr);
	
	
	for(i=0;i<8;++i)
	{
		k = pp[p_id].cubeind[i];
		del[i] = 1.0;
		for(j=0;j<3;++j)
		{del[i]*=  (fabs(pp[p_id].x[j]-grid[k][j])/dx[j])*(fabs(pp[p_id].x[j]-grid[k][j])/dx[j]);
         	 
		}
			
		
		rvphi+= del[i]*meshphi[k];
		
	}	
	for(i=0;i<8;++i)
	{
		k = pp[p_id].cubeind[i];//  printf("% d\n",k);
		//tul00[k]+= m*del[i]*(1.0+3.0*rvphi-rvphi-gamma*gamma*(vmgsqr*ap*ap*rvphi+rvphi))/(ap*ap*ap);
		tuldss[k]+= (vmgsqr*gamma/3.0)*del[i]*(3.0*rvphi-rvphi-gamma*gamma*(vmgsqr*ap*ap*rvphi+rvphi))/(ap*ap*ap);
		psty[k]+= sqrt(vmgsqr)*m*del[i]*(1.0+3.0*rvphi-gamma*gamma*vmgsqr*ap*ap*rvphi)/(ap*ap*ap);
		usty[k]+= m*del[i]*gamma*(-1.0-gamma*gamma)/(6.0*a_t*a_t*Mpl*Mpl*ap);

		 
		
			// printf("pacc  %lf   %lf   %d\n",(ap),tul00[k],p_id);
	}

	
	

	

}


void background()
{ H0 =24.76*(1e-8);
   
   Vamp =1.0;
   int j;
   
   int fail=1;
   ai = 1.0;
   a0 = 1000.0;
   a_t = ai;
   double cpmc = (0.14/(0.68*0.68));
    ommi = (cpmc)*pow((a0/ai),3.0)/(cpmc*a0*a0*a0/(ai*ai*ai) + (1.0-cpmc));
    omdei = 1.0-ommi;
   
  Hi = H0*sqrt(cpmc*a0*a0*a0/(ai*ai*ai) + (1.0-cpmc));
    

    printf("WWWW  %.10lf  Hi %.10lf\n",a0/a_t,Hi);
    

  
}







void initialise()
{
      int l1,l2,r1,r2;

      double cpmc = (0.14/(0.68*0.68));
      int px,py,pz,ci,pgi,j;
      int xcntr[3]={-1,-1,-1};
      double gamma, v, gradmagf;
      a0 = 1000.0;
      ai = 1.0;
      a = ai;
      omdmb= (cpmc)*pow((a0/ai),3.0)/(cpmc*a0*a0*a0/(ai*ai*ai) + (1.0-cpmc));

	dx[0] = 0.001; dx[1] =0.001; dx[2] = 0.001;
        L[0] = dx[0]*((double) (n-1));  L[1] = dx[1]*((double) (n-1));  L[2] = dx[2]*((double) (n-1));


	ini_rand_field();
        
	for(ci = 0;ci <tN; ++ci)
	{
		
		
		if((ci%(n*n))==0)
		 ++xcntr[0];
		if((ci%(n))==0)
		 ++xcntr[1];
		 ++xcntr[2];

		for(j=0;j<3;++j)
		{	p[ci].x[j] =  ((double) rand()/((double) RAND_MAX  ))*L[j];
			

			grid[ci][j] = (xcntr[j]%n)*dx[j];
			
		
			//printf("grid ini  %d  %d  %d %lf\n",ci,j,(xcntr[j]%n),grid[ci][j]);
		}
		
				
		usty[ci]=0.0;
		psty[ci]=0.0;		

		 phi[ci] = ini_phi[ci][0];
		 phi_a[ci] = 0.0;
		
      	}
   
	ini_displace_particle();
	#pragma omp parallel for
	  for(ci=0;ci<tN;++ci)
	  {
	    particle2mesh(p,ci,phi,a);

	    usty[ci]+= 1.0;
	    
	  

	   
	    LAPphi[ci] = 0.0;
	    
	    for(j=0;j<3;++j)
	     {	 
		l1 = (tN+ci-((int)(pow(n,2-j))))%tN;
		l2 = (tN+ci-2*((int)(pow(n,2-j))))%tN;
		r1 = (tN+ci+((int)(pow(n,2-j))))%tN;
		r2 = (tN+ci+2*((int)(pow(n,2-j))))%tN;

		
		
		phi_s[j][ci] = (phi[l2]-8.0*phi[l1]+8.0*phi[r1]-phi[r2])/(12.0*dx[j]); 
		
		
		LAPphi[ci] += (-phi[l2]+16.0*phi[l1]-30.0*phi[ci]+16.0*phi[r1]-phi[r2])/(12.0*dx[j]*dx[j]); 
		

		
		
	     }
  		


	  }
	           
          
	free(ini_vel0); free(ini_vel1); free(ini_vel2);
	printf("Initialization Complete.");


	  

}


int evolve(double aini, double astp)
{
    


    double ommi = omdmb;
    double facb;
    double w;

    int fail = 1,i,j,lcntr,ci;

    double nd = (double) n, jd;  ///Watch out for local vs global for parallelization
    double phiacc,facc,pacc[3],v,gamma,phiavg,phi_aavg,phi_savg[3],fsg,phiold;
    double vmagsqr;
    int anchor[3];
    double lplphi,lplf;
    int l2,l1,r1,r2;
    

    for(a=aini,lcntr=0;((a/ai)<=astp)&&(fail==1);++lcntr)
    { if(lcntr%jprint==0)
	   printf("a  %lf %d\n",a/ai,j);
          
          
	  a_t = Hi*sqrt(ommi*ai*ai*ai/(a)  + (1.0-ommi)*a*a ) ; 
       
        a_tt =  -0.5*ommi*Hi*Hi*ai*ai*ai/(a*a) + (1.0-ommi)*Hi*Hi*a;
         
	  
	  

/////////////////////////////////particle force calculation*****Step 1////////////////////////////////////////////////		 

	 for(ci=0;ci<tN;++ci)
	  {
			vmagsqr = 0.0;	

	
		phiavg = mesh2particle(p,ci,phi);
		
		phi_aavg = mesh2particle(p,ci,phi_a);
		fsg = 0.0;
		for(i=0;i<3;++i)
		{	phi_savg[i] = mesh2particle(p,ci,&phi_s[i][0]);
			
			fsg+= 2.0*p[ci].v[i]*a_t*( phi_savg[i] + phi_savg[i] );
			vmagsqr+=p[ci].v[i]*p[ci].v[i];

		}
			gamma = 1.0/sqrt(1.0-a*a*a_t*a_t*vmagsqr);
		for(i=0;i<3;++i)
		{
			
			pacc[i] = (p[ci].v[i]*a_t*a_t*a_t*vmagsqr*(-2.0*a*a_t*(phiavg+phiavg)-a*a*phi_aavg*a_t+a*a_t)
				 +p[ci].v[i]*a_t*(fsg + phi_aavg*a_t -2.0*a_t/a + 2.0*phi_aavg*a_t -phi_savg[i])
				 -phi_savg[i]/(a))/(a*a) - a_tt*p[ci].v[i]/(a_t*a_t);
			tmpp[ci].x[i] = p[ci].x[i] + 0.5*da*p[ci].v[i];
			tmpp[ci].v[i] = p[ci].v[i] + 0.5*da*pacc[i]; 

		if((tmpp[ci].x[i]>L[i])||(tmpp[ci].x[i]<0.0))
			tmpp[ci].x[i] = fmod(tmpp[ci].x[i]+L[i],L[i]);
		
		
			anchor[i] =  ( n + (int) (tmpp[ci].x[i]/dx[i]))%n;

			
			

	

		}
			tmpp[ci].cubeind[0] = anchor[0]*n*n + anchor[1]*n + anchor[2];
			tmpp[ci].cubeind[1] = ((anchor[0]+1)%n)*n*n + anchor[1]*n + anchor[2];
			tmpp[ci].cubeind[2] = anchor[0]*n*n + ((anchor[1]+1)%n)*n +   anchor[2];
			tmpp[ci].cubeind[3] = anchor[0]*n*n + anchor[1]*n  + ((anchor[2]+1)%n);
			tmpp[ci].cubeind[4] = ((anchor[0]+1)%n)*n*n + ((anchor[1]+1)%n)*n + anchor[2];
			tmpp[ci].cubeind[5] = ((anchor[0]+1)%n)*n*n + anchor[1]*n +   ((anchor[2]+1)%n);
			tmpp[ci].cubeind[6] = anchor[0]*n*n + ((anchor[1]+1)%n)*n +   ((anchor[2]+1)%n);
			tmpp[ci].cubeind[7] = ((anchor[0]+1)%n)*n*n + ((anchor[1]+1)%n)*n + ((anchor[2]+1)%n);
/////////////////////phi acceleration calculation Step 1/////////////////////////////////////////////////////////////////////////////////

		phiacc = (1.0/(a_t*a*a_t*a))*(-2.0*phi[ci]*a_t*a_t - 4.0*a*phi[ci]*a_tt 
					      -a*a*tuldss[ci]/(3.0*Mpl*Mpl))/(a_t*a_t) - 3.0*phi_a[ci]/a -phi_a[ci]/a - a_tt*phi_a[ci]/(a_t*a_t);

		
		tmpphi[ci]  = phi[ci]+0.5*da*phi_a[ci];
		tmpphi_a[ci] = phi_a[ci]+0.5*da*phiacc;
		usty[ci] = 0.0 ;
		psty[ci] = 0.0 ;
		tul00[ci] = 0.0;
		tuldss[ci] = 0.0;
		
		
	  }
		
	  ak = a + 0.5*da;
 
	    a_t = Hi*sqrt(ommi*ai*ai*ai/(ak)  + (1.0-ommi)*ak*ak ) ; 
       
        a_tt =  -0.5*ommi*Hi*Hi*ai*ai*ai/(ak*ak) + (1.0-ommi)*Hi*Hi*ak;
/////////////////////Intermediate Tul calculations and Psi construction//////////////////////////////////////////
	  #pragma omp parallel for
	  for(ci=0;ci<tN;++ci)
	  {
	    particle2mesh(tmpp,ci,tmpphi,ak);

	    usty[ci]+= 1.0;

	    

	    
	    LAPphi[ci] = 0.0;
	 
	    for(j=0;j<3;++j)
	     {	 
		l1 = (tN+ci-((int)(pow(n,2-j))))%tN;
		l2 = (tN+ci-2*((int)(pow(n,2-j))))%tN;
		r1 = (tN+ci+((int)(pow(n,2-j))))%tN;
		r2 = (tN+ci+2*((int)(pow(n,2-j))))%tN;

		
		
		phi_s[j][ci] = (tmpphi[l2]-8.0*tmpphi[l1]+8.0*tmpphi[r1]-tmpphi[r2])/(12.0*dx[j]); 
		
		LAPphi[ci] += (-tmpphi[l2]+16.0*tmpphi[l1]-30.0*tmpphi[ci]+16.0*tmpphi[r1]-tmpphi[r2])/(12.0*dx[j]*dx[j]); 

		
	     }
  	    


	  }
	
	 
	




/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 


////////////////////////////Final Step//////////////////////////////////////////////////////////////////////////////////////
          
 	  
	  

	 

	 for(ci=0;ci<tN;++ci)
	  {
			vmagsqr = 0.0;	

		
		phiavg = mesh2particle(tmpp,ci,tmpphi);
		
		phi_aavg = mesh2particle(tmpp,ci,tmpphi_a);
		fsg = 0.0;
		for(i=0;i<3;++i)
		{	phi_savg[i] = mesh2particle(tmpp,ci,&phi_s[i][0]);
			
			fsg+= 2.0*tmpp[ci].v[i]*a_t*( phi_savg[i] + phi_savg[i] );
			vmagsqr+=tmpp[ci].v[i]*tmpp[ci].v[i];

		}
			gamma = 1.0/sqrt(1.0-ak*ak*a_t*a_t*vmagsqr);
		for(i=0;i<3;++i)
		{
			
			pacc[i] = (tmpp[ci].v[i]*a_t*a_t*a_t*vmagsqr*(-2.0*ak*a_t*(phiavg+phiavg)-ak*ak*phi_aavg*a_t+a*a_t)
				 +tmpp[ci].v[i]*a_t*(fsg + phi_aavg*a_t -2.0*a_t/ak + 2.0*phi_aavg*a_t -phi_savg[i])
				 -phi_savg[i]/(ak))/(ak*ak) - a_tt*tmpp[ci].v[i]/(a_t*a_t);
			p[ci].x[i] = p[ci].x[i] + 0.5*da*p[ci].v[i];
			p[ci].v[i] = p[ci].v[i] + 0.5*da*pacc[i]; 

		if((p[ci].x[i]>L[i])||(p[ci].x[i]<0.0))
			p[ci].x[i] = fmod(p[ci].x[i]+L[i],L[i]);
		


			anchor[i] =  ( n + (int) (tmpp[ci].x[i]/dx[i]))%n; 


			if(isnan(p[ci].x[i] +p[ci].v[i] ))
				fail=0;
		
		}
			p[ci].cubeind[0] = anchor[0]*n*n + anchor[1]*n + anchor[2];
			p[ci].cubeind[1] = ((anchor[0]+1)%n)*n*n + anchor[1]*n + anchor[2];
			p[ci].cubeind[2] = anchor[0]*n*n + ((anchor[1]+1)%n)*n +   anchor[2];
			p[ci].cubeind[3] = anchor[0]*n*n + anchor[1]*n  + ((anchor[2]+1)%n);
			p[ci].cubeind[4] = ((anchor[0]+1)%n)*n*n + ((anchor[1]+1)%n)*n + anchor[2];
			p[ci].cubeind[5] = ((anchor[0]+1)%n)*n*n + anchor[1]*n +   ((anchor[2]+1)%n);
			p[ci].cubeind[6] = anchor[0]*n*n + ((anchor[1]+1)%n)*n +   ((anchor[2]+1)%n);
			p[ci].cubeind[7] = ((anchor[0]+1)%n)*n*n + ((anchor[1]+1)%n)*n + ((anchor[2]+1)%n);
/////////////////////phi  acceleration calculation Final/////////////////////////////////////////////////////////////////////////////////
		
		phiacc = (1.0/(a_t*ak*a_t*ak))*(-2.0*tmpphi[ci]*a_t*a_t - 4.0*ak*tmpphi[ci]*a_tt 
				-ak*ak*tuldss[ci]/(3.0*Mpl*Mpl))/(a_t*a_t) - 3.0*tmpphi_a[ci]/ak -tmpphi_a[ci]/a - a_tt*tmpphi_a[ci]/(a_t*a_t);

		
		
		phi[ci]  = phi[ci]+da*tmpphi_a[ci];
		phi_a[ci] = phi_a[ci]+da*phiacc;

 		if(isnan(phi[ci]+phi_a[ci]))
		fail=0;
		usty[ci] = 0.0;
		psty[ci] = 0.0;
		tul00[ci] = 0.0;
		tuldss[ci] = 0.0;



	}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////Final Tul and Psi recosntruction/////////////////////////////////////////////////////////////
	a = a+ da;	


	  #pragma omp parallel for
	  for(ci=0;ci<tN;++ci)
	  {
	    particle2mesh(p,ci,phi,a);

	    usty[ci]+= 1.0;

	  
	    LAPphi[ci] = 0.0;
	    LAPphi[ci] = 0.0;	
	    for(j=0;j<3;++j)
	     {	 
		l1 = (tN+ci-((int)(pow(n,2-j))))%tN;
		l2 = (tN+ci-2*((int)(pow(n,2-j))))%tN;
		r1 = (tN+ci+((int)(pow(n,2-j))))%tN;
		r2 = (tN+ci+2*((int)(pow(n,2-j))))%tN;

		
		
		phi_s[j][ci] = (phi[l2]-8.0*phi[l1]+8.0*phi[r1]-phi[r2])/(12.0*dx[j]); 
		
		LAPphi[ci] += (-phi[l2]+16.0*phi[l1]-30.0*phi[ci]+16.0*phi[r1]-phi[r2])/(12.0*dx[j]*dx[j]); 

		
	     }
  	    


	  }
	 
   
   

 //   printf("evolve w  %.10lf  Hi %.10lf  %.10lf  %.10lf\n",a_t,a,a0);

    if(fail!=1)
    {printf("fail  %d\n",fail); 
	return(fail);
    }    
	
   }
 return(fail);
}






