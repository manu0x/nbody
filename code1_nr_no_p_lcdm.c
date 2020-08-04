/*########### Valid only if second order terms are ignored everywhere and particles are non-relativistic*/
/*########### For non-relativistic particles we do not calculate tmunu(or use particle2mesh) as tmunu for nr particles do not contribute to phiacc.*/

#include <omp.h>
#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h> 
#include <omp.h>
#include <fenv.h>
#include <time.h>
#include "mt19937ar.c"

#define  n 64

#define tpie  2.0*M_PI

FILE *fppwspctrm;


double G   = 1.0;
double c   = 1.0;
double Mpl ;
double lenfac = 1.0;
double Hb0  ;
double L[3];
int tN;
int fail =1;

double *phi, *phi_a,  *f,*f_a,*tul00,*tuldss,fbdss,fb00;
double phi_s[3][n*n*n],f_s[3][n*n*n],LAPf[n*n*n];
double *tmpphi,  *tmpf,*tmpphi_a, *tmpf_a,m=1.0;
double dx[3];
double density_contrast[n*n*n],ini_density_contrast[n*n*n],ini_phi_potn[n*n*n];




double grid[n*n*n][3];
int ind_grid[n*n*n][3];
double k_grid[n*n*n][3];
int kmagrid[n*n*n],kbins,kbincnt[n*n*n];
double dk; double pwspctrm[n*n*n];
double W_cic[n*n*n],C1_cic_shot[n*n*n];

 


 
  
  

fftw_complex *ini_del;
fftw_complex *F_ini_del;
fftw_complex *F_ini_phi;
fftw_complex *ini_phi;


fftw_plan ini_del_plan;
fftw_plan ini_phi_plan;




//int nic[n*n*n][16];

double  omdmb, omdeb, a, ak, a_t, a_tt, Vamp, ai, a0, da, ak, fb, fb_a,a_zels;
double  lin_phi,lin_phi_a,lin_ini_dcm,lin_ini_phi,lin_dcm,lin_growth,kf; int lin_i;
double lin_phi_zeldo,lin_phi_a_zeldo;
double cpmc = (0.14/(0.68*0.68));
int jprint,jprints;
double Hb0, Hi;

FILE *fpback;
FILE *fpdc;
FILE *fpphi;
FILE *fppwspctrm_dc;
FILE *fppwspctrm_phi;
FILE *fp_fields;
FILE *fplinscale;
FILE *fplin;


void background();
void backlin();




void initialise();
double ini_power_spec(double);
void read_ini_rand_field();
void ini_rand_field();

int evolve(double ,double );
void cal_spectrum(double *,FILE *,int);
void cal_dc();

void write_fields();

void cal_grd_tmunu();


void main()
{       Mpl = 1.0/sqrt(8.0*3.142*G) ;
	Hb0  = 22.04*(1e-5)*lenfac;

        da = 1e-5;
        jprint = (int) (0.001/da);
	jprints = jprint*200;
	
	tN=n*n*n;
        
	printf("qqqjprint %d tN %d  Hb0 %.10lf\n",jprint,n,Hb0); 
	//feenableexcept(FE_DIVBYZERO | FE_ItNVALID | FE_OVERFLOW);
	//feenableexcept(FE_DIVBYZERO | FE_ItNVALID | FE_OVERFLOW);

	fftw_init_threads();
	fftw_plan_with_nthreads(4);

	
	
	
	fpdc  = fopen("lcdm_dc.txt","w");
	fpback  = fopen("lcdm_back.txt","w");
	fppwspctrm_dc  = fopen("lcdm_pwspctrm_dc2.txt","w");
	fppwspctrm_phi  = fopen("lcdm_pwspctrm_phi.txt","w");
	fpphi = fopen("lcdm_phi.txt","w");
	
	fplinscale = fopen("lcdm_linscale.txt","w");
	fplin = fopen("lcdm_lpt.txt","w");


        int i;

       // i = fftw_init_threads();
	//	fftw_plan_with_nthreads(omp_get_max_threads());

	phi = (double *) malloc(n*n*n*sizeof(double)); 
        phi_a = (double *) malloc(n*n*n*sizeof(double)); 

	f = (double *) malloc(n*n*n*sizeof(double)); 
        f_a = (double *) malloc(n*n*n*sizeof(double)); 
	tul00 = (double *) malloc(n*n*n*sizeof(double)); 
        tuldss = (double *) malloc(n*n*n*sizeof(double));

	
        tmpphi_a = (double *) malloc(n*n*n*sizeof(double)); 
	
	
        tmpf_a = (double *) malloc(n*n*n*sizeof(double)); 
 


        F_ini_del = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n*n*n);
	ini_del = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n*n*n);
	F_ini_phi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n*n*n);
	ini_phi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n*n*n);
	
	
	

	//m = (double *) malloc(n*n*n*sizeof(double)); 
	
	
      
	
	background();
	initialise();
	
	
       i = evolve(a_zels,a0/ai);
	printf("ffEvvvolove\n");
       cal_dc();
       cal_spectrum(density_contrast,fppwspctrm_dc,0);
		// cal_spectrum(phi,fppwspctrm_phi,0);
       write_fields();
	
	if(i!=1)
	printf("\nIt's gone...\n");




}









void cal_spectrum(double *spcmesh,FILE *fspwrite,int isini)
{	int i,j;
	double delta_pw;

	fftw_complex *dens_cntrst; fftw_complex *Fdens_cntrst;
	fftw_plan spec_plan;

	dens_cntrst = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n*n*n);
	Fdens_cntrst = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n*n*n);

	for(i=0;i<tN;++i)
	{
		dens_cntrst[i][0] = spcmesh[i]; 
		dens_cntrst[i][1] = 0.0;

		pwspctrm[i] = 0.0;
	}

	fftw_plan_with_nthreads(4);
	spec_plan = fftw_plan_dft_3d(n,n,n, dens_cntrst, Fdens_cntrst, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_execute(spec_plan);
	fftw_free(dens_cntrst);

	if(isini==1)
	printf("isini is 1\n");
	else
	printf("isini is NOT 1\n");


	for(i=0;i<tN;++i)
	{
		//if(isini==1)
		pwspctrm[kmagrid[i]]+=  (Fdens_cntrst[i][1]*Fdens_cntrst[i][1] + Fdens_cntrst[i][0]*Fdens_cntrst[i][0])/(n*n*n);
		//else
		//pwspctrm[kmagrid[i]]+=  ((Fdens_cntrst[i][1]*Fdens_cntrst[i][1] + Fdens_cntrst[i][0]*Fdens_cntrst[i][0]))
		//												/(n*n*n*W_cic[i]*W_cic[i]);

	}
	
	for(i=0;i<=kbins;++i)
	{

		if(kbincnt[i]!=0)
	        {  delta_pw = sqrt(pwspctrm[i]*i*i*i*dk*dk*dk/(2.0*M_PI*M_PI*kbincnt[i]))*ai/a;  

		  fprintf(fspwrite,"%lf\t%lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf\n",
							a/ai,i*dk,pwspctrm[i]/(kbincnt[i]),pwspctrm[i]*ai*ai/(kbincnt[i]*a*a),
							pwspctrm[i]/(kbincnt[i]*lin_growth*lin_growth),delta_pw*ai/a,delta_pw/lin_growth,W_cic[i]);

		}

	

	}

	if(isini==1)
	{
		
		
		
		lin_ini_dcm  = sqrt(pwspctrm[lin_i]/(kbincnt[lin_i]));
		lin_ini_phi = -1.5*omdmb*pow(ai/ai,3.0)*Hi*Hi*lin_ini_dcm/(kf*kf/(ai*ai) +3.0*(a_t/ai)*(a_t/ai) );

		//-(2.0*a*a*a/(3.0*ommi*ai*ai*ai))*( 3.0*lin_phi*a_t*a_t/(a*a)  + kf*kf*lin_phi/(Hi*Hi*a*a) )

		printf("li is %d and lin_ini_dcm is %.20lf lin_ini_phi is %.20lf\n",lin_i,lin_ini_dcm,kf);

	}

	
	fftw_free(Fdens_cntrst);
	fftw_destroy_plan(spec_plan);
	fprintf(fspwrite,"\n\n\n\n");
}


double ini_power_spec(double kamp)
{

	//return(0.0001);
	return(0.00001/(kamp+1e-12));


}




void cal_dc()
{

	int i,j,k,l1,l2,r1,r2;
	double lapphi_loc;

	
	



	


	#pragma omp parallel for private(j,lapphi_loc,l1,l2,r1,r2)
  	for(i=0;i<tN;++i)
    	{	
	  lapphi_loc = 0.0;






	    for(j=0;j<3;++j)
	     {	 
		


		l1 = i + ((n+ind_grid[i][j]-1)%n)*((int)(pow(n,2-j))) - ind_grid[i][j]*((int)(pow(n,2-j)));
		l2 = i + ((n+ind_grid[i][j]-2)%n)*((int)(pow(n,2-j))) - ind_grid[i][j]*((int)(pow(n,2-j)));
		r1 = i + ((n+ind_grid[i][j]+1)%n)*((int)(pow(n,2-j))) - ind_grid[i][j]*((int)(pow(n,2-j)));
		r2 = i + ((n+ind_grid[i][j]+2)%n)*((int)(pow(n,2-j))) - ind_grid[i][j]*((int)(pow(n,2-j)));

		//mm = ind_grid[i][j]*(pow(n,2-j));

		//if((l1>tN)&&(l2<0))
		//printf(":))))))\n");

		
		
		
		
		
		
		
		
		lapphi_loc += (-phi[l2]+16.0*phi[l1]-30.0*phi[i]+16.0*phi[r1]-phi[r2])/(12.0*dx[j]*dx[j]); 
	
		
		
		

		
		

		
		
	     }



		

		density_contrast[i] = (-3.0*a_t*a_t*(phi[i])/(a*a) + lapphi_loc/(a*a) -3.0*a_t*phi_a[i]*a_t/a )/
												(3.0*0.5*omdmb*Hi*Hi*ai*ai*ai/(a*a*a));

		fprintf(fpdc,"%d\t%lf\t%.10lf\t%.10lf\t%.10lf\t%.30lf\t%.30lf\n",i,a/ai,grid[i][0],grid[i][1],
						grid[i][2],ini_density_contrast[i],density_contrast[i]);

		
		
	  
	 	
	
	
	
   }
	



	fprintf(fpdc,"\n\n\n");
	


}






void read_ini_rand_field()
{

	int cnt,i;

	FILE *fpinirand = fopen("initial_rand_field.txt","r");

	for(cnt=0;cnt<tN;++cnt)
	{
		fscanf(fpinirand,"%d\t%lf\t%lf\n",
					&i,&ini_density_contrast[cnt],&ini_phi_potn[cnt]);



	}


}



void ini_rand_field()
{	init_genrand(time(0));
	int i,j,k,ief,jef,kef,cnt,rcnt,rk,ri,rj,maxcnt=0; 
	double ksqr,muk,sigk;
	double a1,a2,b1,b2,a,b;

	FILE *fpinirand = fopen("initial_rand_field.txt","w");


	
	double zdvfac = -(2.0/3.0)*a_t/(cpmc*Hb0*Hb0);


	init_genrand(time(0));
	
	for(i=0;i<n;++i)
	{	if(i<=(n/2))
		ief = i;
		else
		ief = i-n;

		if(i==0)
		ri = 0;
		else
		ri = n-i;

		for(j=0;j<(n);++j)
		{
			if(j<=(n/2))
			jef = j;
			else
			jef = j-n;
			
			if(j==0)
			rj = 0;
			else
			rj = n-j;


			for(k=0;k<=(n/2);++k)
			{   cnt = i*n*n + j*n + k;
				if(maxcnt<cnt)
			    	 maxcnt = cnt;
			   	ksqr = 1.0*
					(((double) ief)*((double) ief)/(dx[0]*dx[0]) 
					+ ((double) jef)*((double) jef)/(dx[1]*dx[1])+ ((double) k)*((double) k)/(dx[2]*dx[2]) ) 
					/(((double) n)*((double) n));
			    sigk  = sqrt(ini_power_spec(sqrt(ksqr)));
			    muk = sigk/sqrt(2.0);
		 	    a1 = genrand_res53();
 			    a2 = genrand_res53(); 
			   // b1 = genrand_res53();
 			  //  b2 = genrand_res53();
			    a = (muk*(sqrt(-2.0*log(a1))*cos(2.0*M_PI*a2)));
			    b = (muk*(sqrt(-2.0*log(a1))*cos(2.0*M_PI*a2)));
				
			    F_ini_del[cnt][0] = a;	F_ini_del[cnt][1] = b;


			    if(ksqr!=0.0)
			    {F_ini_phi[cnt][0] = -1.5*omdmb*Hi*Hi*F_ini_del[cnt][0]/(tpie*tpie*ksqr/(ai*ai) -3.0*Hi*Hi );	
			     F_ini_phi[cnt][1] = -1.5*omdmb*Hi*Hi*F_ini_del[cnt][1]/(tpie*tpie*ksqr/(ai*ai)-3.0*Hi*Hi );

				 fprintf(fplinscale,"%d\t%.20lf\t%.20lf\t%.20lf\n",
					        cnt,ksqr/(ai*ai), -3.0*Hi*Hi,(ksqr/(ai*ai))/(3.0*Hi*Hi) );	
			    }
			  else
			    {F_ini_phi[cnt][0] = 0.0;	
			     F_ini_phi[cnt][1] = 0.0;
		 	    } 
		
			
			   
			 

  		 	 			

			  	if(k==0)
				rk = 0;
				else
				rk = n-k;

				rcnt = ri*n*n + rj*n + rk; 
				if(maxcnt<rcnt)
			    	 maxcnt = rcnt;

				F_ini_del[rcnt][0] = F_ini_del[cnt][0];	 F_ini_del[rcnt][1] = -F_ini_del[cnt][1];
				F_ini_phi[rcnt][0] = F_ini_phi[cnt][0];	 F_ini_phi[rcnt][1] = -F_ini_phi[cnt][1];
				
		
			  


			}


		}


	}



	for(i=0;i<=(n/2);i+=(n/2))
	{
		for(j=0;j<=(n/2);j+=(n/2))
		{
			for(k=0;k<=(n/2);k+=(n/2))
			{

					rcnt = i*n*n + j*n + k;
		
				 F_ini_del[rcnt][1] = 0.0;   F_ini_del[rcnt][0] = 0.0;

					F_ini_phi[rcnt][1] = 0.0;  F_ini_phi[rcnt][0] = 0.0;

				



			}

		


		}

		


	}




	//printf("maxx %d\n",maxcnt);








	fftw_plan_with_nthreads(4);
	ini_del_plan = fftw_plan_dft_3d(n,n,n, F_ini_del, ini_del, FFTW_BACKWARD, FFTW_ESTIMATE);
	
	ini_phi_plan = fftw_plan_dft_3d(n,n,n, F_ini_phi, ini_phi, FFTW_BACKWARD, FFTW_ESTIMATE);
	

	fftw_execute(ini_del_plan);
	fftw_execute(ini_phi_plan);
		

	
	for(cnt=0;cnt<tN;++cnt)
	{
		
		ini_del[cnt][0] = ini_del[cnt][0]/sqrt(n*n*n); ini_del[cnt][1] = ini_del[cnt][1]/sqrt(n*n*n); 
		ini_phi[cnt][0] = ini_phi[cnt][0]/sqrt(n*n*n); ini_phi[cnt][1] = ini_phi[cnt][1]/sqrt(n*n*n);
		

		ini_density_contrast[cnt] = ini_del[cnt][0];
		ini_phi_potn[cnt] = ini_phi[cnt][0];


		fprintf(fpinirand,"%d\t%.20lf\t%.20lf\n",
					cnt,ini_density_contrast[cnt],ini_phi_potn[cnt]);


		

	}
    
	 fftw_free(F_ini_phi);
	 fftw_free(F_ini_del);
	

	 fftw_free(ini_phi);
	 fftw_free(ini_del);
	

	fftw_destroy_plan(ini_del_plan);
	fftw_destroy_plan(ini_phi_plan);
	
	printf("Generated initial Gaussian Random field  %d\n",maxcnt);
	
}









void cal_grd_tmunu(int k)
{
	int ci,l1,l2,r1,r2,j,Vvl,V_fvl;

 	if(k==1)
	{
	#pragma omp parallel for private(j,l1,l2,r1,r2)
	  for(ci=0;ci<tN;++ci)
	   {
	  //  particle2mesh(tmpp,ci,tmpphi,a);

	    
	    

	   
	   // LAPphi[ci] = 0.0;
	    
	  
	    
	     for(j=0;j<3;++j)
	     {	 
		


		l1 = ci + ((n+ind_grid[ci][j]-1)%n)*((int)(pow(n,2-j))) - ind_grid[ci][j]*((int)(pow(n,2-j)));
		l2 = ci + ((n+ind_grid[ci][j]-2)%n)*((int)(pow(n,2-j))) - ind_grid[ci][j]*((int)(pow(n,2-j)));
		r1 = ci + ((n+ind_grid[ci][j]+1)%n)*((int)(pow(n,2-j))) - ind_grid[ci][j]*((int)(pow(n,2-j)));
		r2 = ci + ((n+ind_grid[ci][j]+2)%n)*((int)(pow(n,2-j))) - ind_grid[ci][j]*((int)(pow(n,2-j)));
		
		
		phi_s[j][ci] = (phi[l2]-8.0*phi[l1]+8.0*phi[r1]-phi[r2])/(12.0*dx[j]);
		//f_s[j][ci] = (tmpf[l2]-8.0*tmpf[l1]+8.0*tmpf[r1]-tmpf[r2])/(12.0*dx[j]); 
		
		
		//LAPphi[ci] += (-tmpphi[l2]+16.0*tmpphi[l1]-30.0*tmpphi[ci]+16.0*tmpphi[r1]-tmpphi[r2])/(12.0*dx[j]*dx[j]);
		

		
		

		
		
	     }
  		

		
		

		
		//tul00[ci]+= (Vvl + 0.5*tmpf_a[ci]*tmpf_a[ci]*a_t*a_t*(1.0-2.0*phi[ci])) -fb00 ;
		//tuldss[ci]+= 3.0*((Vvl - 0.5*tmpf_a[ci]*tmpf_a[ci]*a_t*a_t*(1.0-2.0*phi[ci])) -fbdss) ;



	  }
	}




	else
	{ 
	// #pragma omp parallel for private(j,l1,l2,r1,r2)
	  for(ci=0;ci<tN;++ci)
	   {
	   //particle2mesh(p,ci,phi,a);

	    

	   
	    //LAPphi[ci] = 0.0;
	    LAPf[ci] = 0.0;
	  	
	    
	     for(j=0;j<3;++j)
	     {	 
		
		

		l1 = ci + ((n+ind_grid[ci][j]-1)%n)*((int)(pow(n,2-j))) - ind_grid[ci][j]*((int)(pow(n,2-j)));
		l2 = ci + ((n+ind_grid[ci][j]-2)%n)*((int)(pow(n,2-j))) - ind_grid[ci][j]*((int)(pow(n,2-j)));
		r1 = ci + ((n+ind_grid[ci][j]+1)%n)*((int)(pow(n,2-j))) - ind_grid[ci][j]*((int)(pow(n,2-j)));
		r2 = ci + ((n+ind_grid[ci][j]+2)%n)*((int)(pow(n,2-j))) - ind_grid[ci][j]*((int)(pow(n,2-j)));
		
		
		phi_s[j][ci] = (phi[l2]-8.0*phi[l1]+8.0*phi[r1]-phi[r2])/(12.0*dx[j]);
		//f_s[j][ci] = (f[l2]-8.0*f[l1]+8.0*f[r1]-f[r2])/(12.0*dx[j]); 
		
		
		//LAPphi[ci] += (-phi[l2]+16.0*phi[l1]-30.0*phi[ci]+16.0*phi[r1]-phi[r2])/(12.0*dx[j]*dx[j]);
		LAPf[ci] += (-f[l2]+16.0*f[l1]-30.0*f[ci]+16.0*f[r1]-f[r2])/(12.0*dx[j]*dx[j]); 

		
		

		
		
	     }


		


		
		
  		


	  }
	}



}







void background()
{ 

   Vamp =1.0;
   int j;
   double Vvl,V_fvl,w,facb;
   double fbk,fb_ak,fbk1,fb_ak1,fbk2,fb_ak2,fbk3,fb_ak3,fbk4,fb_ak4;
   
  
   
   int fail=1,zs_check=0;
   ai = 0.001;
   a0 = 1.0;
   a_t = ai;


  

 Hi = Hb0*sqrt(cpmc*a0*a0*a0/(ai*ai*ai) + (1.0-cpmc));
 
    

    printf("H  %.10lf  Hi %.10lf\n",Hb0,Hi);
    

  
}




void backlin()
{ 

   
   int j;
   
   double lin_phiac1,lin_phiac2,lin_phi_ak;
   
  
   
   int fail=1,zs_check=0;
   ai = 0.001;
   a0 = 1.0;
   a_t = ai;


    double ommi = (cpmc)*pow((a0/ai),3.0)/(cpmc*a0*a0*a0/(ai*ai*ai) + (1.0-cpmc));
    double omdei = 1.0-ommi;
    double omfb;




	
		

		lin_phi = 1.0;
		lin_phi_a =  0.0;
   		
 





	



	
    for(j=0,a=ai;a<=a0&&(fail>0);a+=da,++j)
    {   
	
       
	a_t = sqrt(ommi*ai*ai*ai/(a)  + (1.0-ommi)*a*a ) ; 
       
        a_tt =  -0.5*ommi*ai*ai*ai/(a*a) + (1.0-ommi)*a;
	
	if((a>=a_zels)&&(zs_check==0))

	{
 		

		lin_phi_zeldo = lin_phi;
		

		lin_phi_a_zeldo = lin_phi_a;
		
		zs_check=1;

		
        }



	
	

		lin_dcm = -(2.0*a*a*a/(3.0*ommi*ai*ai*ai))*( 3.0*lin_phi*a_t*a_t/(a*a) + 3.0*lin_phi_a*a_t*a_t/(a) + kf*kf*lin_phi/(Hi*Hi*a*a) );
		//printf("\nlin_dcm %lf inidcm  %lf  \n",lin_dcm*lin_ini_phi,lin_ini_dcm);




		 lin_phiac1 = -4.0*lin_phi_a/a - (2.0*a_tt/(a*a_t*a_t) + 1.0/(a*a))*lin_phi - a_tt*lin_phi_a/(a_t*a_t);

		
     		 lin_phi = lin_phi + lin_phi_a*da + 0.5*lin_phiac1*da*da;
    		 lin_phi_ak = lin_phi_a + lin_phiac1*da;

		 

		


		 
		ak = a+da;


	


	

	if(j%jprint==0)
	fprintf(fpback,"%lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\n",a/ai,ommi*ai*ai*ai/(a*a_t*a_t),a_tt,lin_phi,lin_dcm*lin_ini_phi/lin_ini_dcm);


       
      		a_t = sqrt(ommi*ai*ai*ai/(ak)  + (1.0-ommi)*ak*ak ) ; 
       
        	a_tt =  -0.5*ommi*ai*ai*ai/(ak*ak) + (1.0-ommi)*ak;
      
      

		
	

	
		lin_phiac2 = -4.0*lin_phi_ak/ak - (2.0*a_tt/(ak*a_t*a_t) + 1.0/(ak*ak))*lin_phi  - a_tt*lin_phi_ak/(a_t*a_t);

		
       		lin_phi_a = lin_phi_a + (lin_phiac1+lin_phiac2)*0.5*da;
       



	


	


         
    }
    
   
 	fprintf(fpback,"\n\n\n");


   
    
   
 
}









void write_fields()
{
	int i;

	char name_p[20],name_f[20];
	double f_dc,f_prsr, f_denst,back_f_denst,zaw;
	zaw = a0/a - 1.0;

	
	snprintf(name_f,20,"lcd_fields_z_%lf",zaw);


	fp_fields = fopen(name_f,"w");


	for(i=0;i<tN;++i)
	{
		

		fprintf(fp_fields,"%d\t%lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf\n",
					i,a/ai,grid[i][0],grid[i][1],grid[i][2],density_contrast[i],phi[i],phi_a[i]*a_t);

		

	}

	fclose(fp_fields);
	

}




void initialise()
{
      int l1,l2,r1,r2;

    
      
      int xcntr[3]={-1,-1,-1},anchor[3],ci,j;
      double gamma, v, gradmagf;
      double ktmp,maxkmagsqr = 0.0;
      double wktmp,shtmp;
      a0 = 1.00;
      ai = 0.001;
      a = ai;
      omdmb= (cpmc)*pow((a0/ai),3.0)/(cpmc*a0*a0*a0/(ai*ai*ai) + (1.0-cpmc));
      printf("omdmb  %.10lf\n",omdmb);
      double ommi = omdmb;

     a_t=Hi*ai;

	lin_i = 10;

	kf = tpie*lenfac/(64.0);

	dx[0] = 1.0; dx[1] = 1.0; dx[2] = 1.0;
        L[0] = dx[0]*((double) (n));  L[1] = dx[1]*((double) (n));  L[2] = dx[2]*((double) (n));
	dk = 0.01/dx[0]; kbins = 0;

	//ini_rand_field();
	read_ini_rand_field();
        
	for(ci = 0;ci <tN; ++ci)
	{
		kbincnt[ci]=0;
	}
	
	for(ci = 0;ci <tN; ++ci)
	{
		
		
		if((ci%(n*n))==0)
		 ++xcntr[0];
		if((ci%(n))==0)
		 ++xcntr[1];
		 ++xcntr[2];
		ktmp=0.0;
		wktmp = 1.0;
		shtmp = 1.0;
		for(j=0;j<3;++j)
		{	//
			grid[ci][j] = ((double)(xcntr[j]%n))*dx[j];

			ind_grid[ci][j] = xcntr[j]%n;

			
			
			//anchor[j] =   (int) (p[ci].x[j]/dx[j]); 

			if((xcntr[j]%n)<=(n/2))
				{
					
				 k_grid[ci][j] = (xcntr[j]%n)/L[j];

				  ktmp+= k_grid[ci][j]*k_grid[ci][j];

					if((xcntr[j]%n)==0)
					wktmp*=1.0;
					else
					wktmp*=(sin((xcntr[j]%n)*dx[j]*0.5/L[j])/((xcntr[j]%n)*dx[j]*0.5/L[j]));

					shtmp*=(1.0   -  (2.0/3.0)*sin((xcntr[j]%n)*dx[j]*0.5/L[j])*sin((xcntr[j]%n)*dx[j]*0.5/L[j]));	

				}
			else
				{ 
				 k_grid[ci][j] = ((xcntr[j]%n)-n)/L[j];

				 ktmp+= k_grid[ci][j]*k_grid[ci][j];

				wktmp*=(sin(((xcntr[j]%n)-n)*dx[j]*0.5/L[j])/(((xcntr[j]%n)-n)*dx[j]*0.5/L[j]));	
				shtmp*=(1.0 - (2.0/3.0)*sin(((xcntr[j]%n)-n)*dx[j]*0.5/L[j])*sin(((xcntr[j]%n)-n)*dx[j]*0.5/L[j]));
				  
				}
		
			 
			
			
			//printf("grid ini  %d  %d  %d %lf\n",ci,j,(xcntr[j]%n),grid[ci][j]);
			//if(xcntr[j]>n)
			//printf("Alert %d  %d\n",j,xcntr[j]);


		}
		
		
		


		W_cic[ci] = wktmp*wktmp;
		C1_cic_shot[ci] =  shtmp;	
		
	
			
		if(ktmp>maxkmagsqr)
		maxkmagsqr = (ktmp);

		kmagrid[ci] = (int)(sqrt(ktmp)/(dk));
		 //printf("yo  %d  %lf\n",kmagrid[ci],sqrt(ktmp));
		++kbincnt[kmagrid[ci]];

		if(kmagrid[ci]>kbins)
		kbins=kmagrid[ci];
		
			

		 
		 phi[ci] = ini_phi_potn[ci];
		 phi_a[ci] = 0.0;
		 
		 

		
		
      	}





  
	cal_spectrum(ini_density_contrast,fppwspctrm_dc,1);

  
	a_zels = ai;
	backlin();

	

	a = a_zels;

	

	


	for(ci=0;ci<tN;++ci)
	  {

		
	
	  
		
		

		tul00[ci]= 0.0;
		tuldss[ci]=0.0;
		
			
		

	}


	
	           
          
	

	
	


	  a_t = Hi*sqrt(ommi*ai*ai*ai/(a)  + (1.0-ommi)*a*a ) ; 
       
        a_tt =  -0.5*ommi*Hi*Hi*ai*ai*ai/(a*a) + (1.0-ommi)*Hi*Hi*a;
	cal_dc();

	cal_grd_tmunu(0);

	cal_spectrum(density_contrast,fppwspctrm_dc,0);

	printf("Initialization Complete.\n");
	printf("\nK details:\n	dk is %lf  per MPc",dk/lenfac);
	printf("\n Nyquist Wavenumber is %lf",M_PI/dx[0]);
	printf("\n	Min k_mag is %lf per MPc:: corr lmbda is %.10lf MPc",1.0/(dx[0]*lenfac*((double) n)),dx[0]*lenfac*((double) n));
	printf("\n	Max k_mag is %lf per MPc:: corr lmbda is %.10lf Mpc",sqrt(maxkmagsqr)/lenfac,lenfac/sqrt(maxkmagsqr));
	printf("\n	kbins is %d\n",kbins);

	printf("\nLengthscales:");
	printf("\n	Grid Length is %.5lf MPc",dx[0]*lenfac*((double) n));
	printf("\n	dx is %.10lf MPc\n",dx[0]*lenfac);

	printf("\n Linear theory kf is %lf kl %lf Mpc %lf\n",kf,tpie*lenfac/kf,lin_phi);
	

	  

}


int evolve(double aini, double astp)
{
    printf("Evvvolove\n");


    double ommi = omdmb;
    double facb1,facb2,lin_delfac1,lin_phiac1,lin_phiac2,lin_phi_ak;
    double w;

    int i,j,lcntr,ci;
     
     ///Watch out for local vs global for parallelization
    double phiacc1[n*n*n],phiacc2[n*n*n];



    Vamp = Vamp*Hi*Hi; 
  
   

  for(a=aini,lcntr=0;((a/ai)<=astp)&&(fail==1);++lcntr)
    { //if(lcntr%jprint==0)
	   
          
      a_t = Hi*sqrt(ommi*ai*ai*ai/(a)  + (1.0-ommi)*a*a ) ; 
       
        a_tt =  -0.5*ommi*Hi*Hi*ai*ai*ai/(a*a) + (1.0-ommi)*Hi*Hi*a;


	lin_dcm = -(2.0*a*a*a/(3.0*Hi*Hi*ommi*ai*ai*ai))*( 3.0*lin_phi*a_t*a_t/(a*a) + 3.0*lin_phi_a*a_t*a_t/(a) + kf*kf*lin_phi/(a*a) ) ;

	lin_growth = lin_dcm*lin_ini_phi/lin_ini_dcm;



	

	
	lin_phiac1 = -4.0*lin_phi_a/a - (2.0*a_tt/(a*a_t*a_t) + 1.0/(a*a))*lin_phi  - a_tt*lin_phi_a/(a_t*a_t);

	

     
      lin_phi = lin_phi + lin_phi_a*da + 0.5*lin_phiac1*da*da;
      lin_phi_ak = lin_phi_a + lin_phiac1*da;


         
         
	  if(lcntr%jprint==0)
	  { 
	

		fprintf(fpback,"%lf\t%.10lf\t%.10lf\n",a/ai,ommi*ai*ai*ai*Hi*Hi/(a*a_t*a_t),a_tt);
		fprintf(fplin,"%lf\t%.20lf\t%.20lf\n",a/ai,lin_growth,a/ai);

		printf("a  %lf %.10lf  %.10lf\n",a,ommi*ai*ai*ai*Hi*Hi/(a*a_t*a_t),a);

	
		}

	
	if((lcntr%jprints==0))
	   { printf("printing..\n");

		 cal_dc();
      		 cal_spectrum(density_contrast,fppwspctrm_dc,0);
		// cal_spectrum(phi,fppwspctrm_phi,0);
		 write_fields();


	  }

/////////////////////////////////particle force calculation*****Step 1////////////////////////////////////////////////		 
	#pragma omp parallel for private(i)
	 for(ci=0;ci<tN;++ci)
	  {
		
/////////////////////phi acceleration calculation Step 1/////////////////////////////////////////////////////////////////////////////////

		phiacc1[ci] = (1.0/(a_t*a*a_t*a))*(- 2.0*a*phi[ci]*a_tt 
					      -a*a*tuldss[ci]/(6.0*Mpl*Mpl))  -phi[ci]/(a*a) 
						- 3.0*phi_a[ci]/a -phi_a[ci]/a - a_tt*phi_a[ci]/(a_t*a_t);

		
		
		

		//phiacc = (1.0/(2.0*a_t*a*a_t*a))*(-2.0*phi[ci]*a_t*a_t - 4.0*a*phi[ci]*a_tt 
			//		      -a*a*tuldss[ci]/(3.0*Mpl*Mpl)) - 3.0*phi_a[ci]/a -phi_a[ci]/a - a_tt*phi_a[ci]/(a_t*a_t);
		
		
		phi[ci]  = phi[ci]+da*phi_a[ci]+0.5*da*da*phiacc1[ci];
		tmpphi_a[ci] = phi_a[ci]+da*phiacc1[ci];

		


		

		
		tul00[ci] = 0.0 ;
		tuldss[ci] = 0.0;
		
		
	  }

		//fprintf(fp_particles,"\n\n\n");
		
	 



		
	  ak = a + da;

	 a_t = Hi*sqrt(ommi*ai*ai*ai/(ak)  + (1.0-ommi)*ak*ak ) ; 
       
        a_tt =  -0.5*ommi*Hi*Hi*ai*ai*ai/(ak*ak) + (1.0-ommi)*Hi*Hi*ak;


	lin_phiac2 = -4.0*lin_phi_ak/ak - (2.0*a_tt/(ak*a_t*a_t) + 1.0/(ak*ak))*lin_phi  - a_tt*lin_phi_ak/(a_t*a_t);

	

       
       lin_phi_a = lin_phi_a + (lin_phiac1+lin_phiac2)*0.5*da;
 
	   
/////////////////////Intermediate Tul calculations and psi construction//////////////////////////////////////////


	cal_grd_tmunu(1);
	
	



		
 
	




/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 

	


       

////////////////////////////Final Step//////////////////////////////////////////////////////////////////////////////////////
          
 	  
	  

	 
	#pragma omp parallel for private(i)
	 for(ci=0;ci<tN;++ci)
	  {
			

			
/////////////////////phi  acceleration calculation Final/////////////////////////////////////////////////////////////////////////////////
		
		phiacc2[ci] = (1.0/(a_t*ak*a_t*ak))*(- 2.0*ak*phi[ci]*a_tt 
					      -ak*ak*tuldss[ci]/(6.0*Mpl*Mpl))  -phi[ci]/(ak*ak) 
						- 3.0*tmpphi_a[ci]/ak -tmpphi_a[ci]/ak - a_tt*tmpphi_a[ci]/(a_t*a_t);


		

		
		//phiacc = (1.0/(2.0*a_t*ak*a_t*ak))*(-2.0*tmpphi[ci]*a_t*a_t - 4.0*ak*tmpphi[ci]*a_tt 
		//		-ak*ak*tuldss[ci]/(3.0*Mpl*Mpl)) - 3.0*tmpphi_a[ci]/ak -tmpphi_a[ci]/ak - a_tt*tmpphi_a[ci]/(a_t*a_t);
		

		
		
		phi_a[ci] = phi_a[ci]+0.5*da*(phiacc1[ci]+phiacc2[ci]);
		

 		if(isnan(phi[ci]+phi_a[ci]))
		fail=0;
		
		tul00[ci] = 0.0 ;
		tuldss[ci] = 0.0;


	}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////Final Tul and phi recosntruction/////////////////////////////////////////////////////////////
	 a = a+da;		


        a_t = Hi*sqrt(ommi*ai*ai*ai/(a)  + (1.0-ommi)*a*a ) ; 
       
        a_tt =  -0.5*ommi*Hi*Hi*ai*ai*ai/(a*a) + (1.0-ommi)*Hi*Hi*a;

	

	


	

	cal_grd_tmunu(0);
	
		
	


	
	
	

     

 //   printf("evolve w  %.10lf  Hi %.10lf  %.10lf  %.10lf\n",a_t,a,a0);

    if(fail!=1)
    {//printf("fail  %d %d  %lf\n",fail,lcntr,a); 
	return(fail);
    }    
	
  }
 return(fail);
}






