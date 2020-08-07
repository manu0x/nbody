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
double lenfac =1.0;
double Hb0  ;
double L[3];
double n3sqrt;
int tN;
int fail =1;

double *phi, *phi_a,  *f,*f_a,*tul00,*tuldss,fbdss,fb00;
double phi_s[3][n*n*n],f_s[3][n*n*n],LAPf[n*n*n];
double *tmpphi,  *tmpf,*tmpphi_a, *tmpf_a,m=1.0;
double dx[3],d1[3],d2[3];
double density_contrast[n*n*n],ini_density_contrast[n*n*n],ini_phi_potn[n*n*n];
struct particle
	{	
		double x[3];
		double v[3];
		
		
		int 	cubeind[8];	

	};



struct particle p[n*n*n],tmpp[n*n*n];
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





fftw_plan scf_plan_f;
fftw_plan scf_plan_b;

fftw_complex *scf_rhs;
fftw_complex *scf_rhs_ft;





//int nic[n*n*n][16];

double  omdmbini, omdeb, a, ak, a_t, a_tt, Vamp, ai, a0, da, fb, fb_a,a_zels;
double  lin_delf,lin_phi,lin_delf_a,lin_phi_a,lin_ini_dcm,lin_ini_phi,lin_dcm,lin_dcf,lin_growth,kf; int lin_i;
double fb_zeldo,fb_a_zeldo,lin_phi_zeldo,lin_phi_a_zeldo,lin_delf_zeldo,lin_delf_a_zeldo;
double cpmc = (0.14/(0.68*0.68));
int jprint,jprints;
double Hb0, Hi;

FILE *fpback;
FILE *fp_particles;
FILE *fpdc;
FILE *fpphi;
FILE *fppwspctrm_dc;
FILE *fppwspctrm_phi;
FILE *fp_fields;
FILE *fplinscale;
FILE *fplin;

void background(int);




void initialise();
double ini_power_spec(double);
void read_ini_rand_field();
void ini_rand_field();
int evolve(double ,double );
void cal_spectrum(double *,FILE *,int);
void cal_dc();
void write_fields();
double V(double);
double V_f(double);
double V_ff(double);
void cal_grd_tmunu();


void main()
{       Mpl = 1.0/sqrt(8.0*3.142*G) ;
	Hb0  = 22.04*(1e-5)*lenfac;

        da = 1e-5;
        jprint = (int) (0.001/da);
	jprints = jprint*200;
	
	tN=n*n*n;
	n3sqrt = sqrt((double) tN);
        
	printf("qqqjprint %d tN %d  Hb0 %.10lf\n",jprint,n,Hb0); 
	//feenableexcept(FE_DIVBYZERO | FE_ItNVALID | FE_OVERFLOW);
	//feenableexcept(FE_DIVBYZERO | FE_ItNVALID | FE_OVERFLOW);

	fftw_init_threads();
	fftw_plan_with_nthreads(128);

	
	
	
	fpdc  = fopen("lin_dc.txt","w");
	fpback  = fopen("lin_back.txt","w");
	fppwspctrm_dc  = fopen("lin_pwspctrm_dc2.txt","w");
	fppwspctrm_phi  = fopen("lin_pwspctrm_phi.txt","w");
	fpphi = fopen("lin_phi.txt","w");
	
	fplinscale = fopen("lin_linscale.txt","w");
	fplin = fopen("lin_lpt.txt","w");


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
	


	
	scf_rhs = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n*n*n);
	scf_rhs_ft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * n*n*n);

	

	scf_plan_f = fftw_plan_dft_3d(n,n,n, scf_rhs, scf_rhs_ft, FFTW_FORWARD, FFTW_ESTIMATE);
	scf_plan_b = fftw_plan_dft_3d(n,n,n, scf_rhs_ft, scf_rhs, FFTW_BACKWARD, FFTW_ESTIMATE);

	

	//m = (double *) malloc(n*n*n*sizeof(double)); 
	
	
      
	
	background(0);
	initialise();
	
	
       i = evolve(a_zels,a0/ai);
	printf("ffEvvvoloved...\n");
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
	        {  delta_pw = sqrt(pwspctrm[i]*i*i*i*dk*dk*dk/(2.0*M_PI*M_PI*kbincnt[i]));  

		   fprintf(fspwrite,"%lf\t%lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf\n",
							a/ai,i*dk,pwspctrm[i]/(kbincnt[i]),pwspctrm[i]*ai*ai/(kbincnt[i]*a*a),
							pwspctrm[i]/(kbincnt[i]*lin_growth*lin_growth),delta_pw*ai/a,delta_pw/lin_growth,W_cic[i]);

		}

	

	}
	
	if(isini==1)
	{
		
		
		
		lin_ini_dcm  = sqrt(pwspctrm[lin_i]/(kbincnt[lin_i]));
		lin_ini_phi = -1.5*omdmbini*pow(ai/ai,3.0)*Hi*Hi*lin_ini_dcm/(kf*kf/(ai*ai) +3.0*(a_t/ai)*(a_t/ai) );

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
	double lapphi_loc,t00f_loc,Vvl;
	

	
	



	


	#pragma omp parallel for private(j,lapphi_loc,t00f_loc,l1,l2,r1,r2,Vvl)
  	for(i=0;i<tN;++i)
    	{	
	  lapphi_loc = 0.0;
	  t00f_loc = 0.0;





	    for(j=0;j<3;++j)
	     {	 
		


		l1 = i + ((n+ind_grid[i][j]-1)%n)*((int)(pow(n,2-j))) - ind_grid[i][j]*((int)(pow(n,2-j)));
		l2 = i + ((n+ind_grid[i][j]-2)%n)*((int)(pow(n,2-j))) - ind_grid[i][j]*((int)(pow(n,2-j)));
		r1 = i + ((n+ind_grid[i][j]+1)%n)*((int)(pow(n,2-j))) - ind_grid[i][j]*((int)(pow(n,2-j)));
		r2 = i + ((n+ind_grid[i][j]+2)%n)*((int)(pow(n,2-j))) - ind_grid[i][j]*((int)(pow(n,2-j)));

		//mm = ind_grid[i][j]*(pow(n,2-j));

		//if((l1>tN)&&(l2<0))
		//printf(":))))))\n");

		
		
		
		
		
		
		
		
		lapphi_loc += (-phi[l2]+16.0*phi[l1]-30.0*phi[i]+16.0*phi[r1]-phi[r2])/(d2[j]); 
	
		
		
		

		
		

		
		
	     }


		Vvl = V(f[i]);


		t00f_loc+=Vvl + 0.5*f_a[i]*f_a[i]*a_t*a_t*(1.0-2.0*(phi[i])) - fb00;

		density_contrast[i] = (-3.0*a_t*a_t*(phi[i])/(a*a) + lapphi_loc/(a*a) -3.0*a_t*phi_a[i]*a_t/a -0.5*t00f_loc/(Mpl*Mpl))/
												(3.0*0.5*omdmbini*Hi*Hi*ai*ai*ai/(a*a*a));

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
			    {F_ini_phi[cnt][0] = -1.5*omdmbini*Hi*Hi*F_ini_del[cnt][0]/(tpie*tpie*ksqr/(ai*ai) + 3.0*Hi*Hi );	
			     F_ini_phi[cnt][1] = -1.5*omdmbini*Hi*Hi*F_ini_del[cnt][1]/(tpie*tpie*ksqr/(ai*ai) + 3.0*Hi*Hi );

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















double V(double fff)
{
   return(Vamp*(fff)*(fff));
}


double V_f(double fff)
{
   return(2.0*Vamp*(fff));


}

double V_ff(double fff)
{
   return(2.0*Vamp);


}


void background(int bk)
{ 

   Vamp =1.0;
   int j;
   double Vvl,V_fvl,V_ffvl,w,facb;
   double fbk,fb_ak,fbk1,fb_ak1,fbk2,fb_ak2,fbk3,fb_ak3,fbk4,fb_ak4;
   double lin_delfac1,lin_delfac2,lin_phiac1,lin_phiac2,lin_delf_ak,lin_phi_ak;
   
  
   
   int fail=1,zs_check=0;
   ai = 0.001;
   a0 = 1.0;
   a_t = ai;


    double ommi = (cpmc)*pow((a0/ai),3.0)/(cpmc*a0*a0*a0/(ai*ai*ai) + (1.0-cpmc));
    double omdei = 1.0-ommi;
    double omfb;

   fb = 1.0;
   
  fb_a =0.0/a_t;
     Vamp = 3.0*Mpl*Mpl*omdei*c/V(fb); 


	if(bk==1)
	{
		//lin_i = 50;
	
		//kf = tpie*lin_i*dk;

		

		lin_phi = 1.0;
   		lin_delf = 0.0;
   		lin_phi_a =  0.0;
   		lin_delf_a = 0.0/ini_phi_potn[lin_i];
 



	}

	



	
    for(j=0,a=ai;a<=a0&&(fail>0);a+=da,++j)
    {   Vvl = V(fb);
  	V_fvl = V_f(fb);
	a_t = sqrt((ommi*ai*ai*ai/a  + (1.0/(Mpl*Mpl))*a*a*Vvl/(3.0*c)) / ( 1.0 - (1.0/(Mpl*Mpl))*a*a*fb_a*fb_a/(6.0*c*c*c))) ;
        a_tt = -0.5*ommi*ai*ai*ai/(a*a) - (1.0/(Mpl*Mpl*c))*a*(fb_a*fb_a*a_t*a_t - Vvl)/3.0;
        facb = -V_fvl*c*c/(a_t*a_t) - 3.0*fb_a/a - a_tt*fb_a/(a_t*a_t);
        w = (fb_a*fb_a*a_t*a_t/(2.0*c*c) - Vvl)/(fb_a*fb_a*a_t*a_t/(2.0*c*c) + Vvl);


	omfb = (1.0/(3.0*c*c*c*Mpl*Mpl))*(fb_a*fb_a*a_t*a_t/(2.0*c*c) + Vvl); 
	
	if((a>=a_zels)&&(zs_check==0))

	{
 		fb_zeldo = fb;

		fb_a_zeldo = fb_a;

		lin_phi_zeldo = lin_phi;
		lin_delf_zeldo = lin_delf;

		lin_phi_a_zeldo = lin_phi_a;
		lin_delf_a_zeldo = lin_delf_a;

		zs_check=1;

		printf("fb_z  %lf\n fb_a_z  %lf\n",fb_zeldo,fb_a_zeldo);
        }



	
	if(bk==1)
	{

		 V_ffvl =  V_ff(fb);
		

		lin_dcf = (V_fvl*lin_delf - lin_phi*fb_a*a_t*fb_a*a_t + fb_a*a_t*lin_delf_a*a_t)/(0.5*fb_a*a_t*fb_a*a_t + Vvl);
		lin_dcm = -(2.0*a*a*a/(3.0*ommi*ai*ai*ai))*( 3.0*lin_phi*a_t*a_t/(a*a) + 3.0*lin_phi_a*a_t*a_t/(a) + kf*kf*lin_phi/(Hi*Hi*a*a) )
											 - omfb*lin_dcf*a*a*a/(ommi*ai*ai*ai) ;
		if(j==0)
		printf("\nlin_dcm %lf inidcm  %lf  \n",lin_dcm*lin_ini_phi,lin_ini_dcm);



		 lin_delfac1 =  -3.0*lin_delf_a/a - kf*kf*lin_delf/(a*a*(a_t*Hi)*(a_t*Hi)) - 2.0*lin_phi*V_fvl/(a_t*a_t) 
				+ 4.0*lin_phi_a*fb_a- V_ffvl*lin_delf/(a_t*a_t) - a_tt*lin_delf_a/(a_t*a_t);

		 lin_phiac1 = -4.0*lin_phi_a/a - (2.0*a_tt/(a*a_t*a_t) + 1.0/(a*a))*lin_phi - (0.5/(Mpl*Mpl))*(lin_delf*V_fvl/(a_t*a_t) 
			+ lin_phi*fb_a*fb_a - fb_a*lin_delf_a) - a_tt*lin_phi_a/(a_t*a_t);

		 lin_delf = lin_delf + lin_delf_a*da + 0.5*lin_delfac1*da*da;
     		 lin_delf_ak = lin_delf_a + lin_delfac1*da;
     		 lin_phi = lin_phi + lin_phi_a*da + 0.5*lin_phiac1*da*da;
    		 lin_phi_ak = lin_phi_a + lin_phiac1*da;

		 

		if(j<3)
		printf("back_lpt  delfac1  %lf phiac1  %lf\n",lin_delfac1,lin_phiac1);


		 



	}


	

	if(j%jprint==0)
	fprintf(fpback,"%lf\t%.10lf\t%.10lf\t%.10lf\t%.10lf\n",a/ai,ommi*ai*ai*ai/(a*a_t*a_t),omfb*a*a/(a_t*a_t),lin_phi,lin_dcm*lin_ini_phi/lin_ini_dcm);


       
      
      
        ak = a+0.5*da;
        fb_ak1 =  facb*da;
        fbk1 = da*fb_a; 
      
        fb_ak = fb_a + 0.5*fb_ak1;
        fbk = fb + 0.5*fbk1; 
        

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


	if(bk==1)
	{
		V_fvl = V_f(fb);
		V_ffvl =  V_ff(fb);

		 lin_delfac2 =  -3.0*lin_delf_ak/ak - kf*kf*lin_delf/(ak*ak*(a_t*Hi)*(a_t*Hi)) - 2.0*lin_phi*V_fvl/(a_t*a_t) 
				+ 4.0*lin_phi_ak*fb_ak- V_ffvl*lin_delf/(a_t*a_t) - a_tt*lin_delf_ak/(a_t*a_t);

	

	
		lin_phiac2 = -4.0*lin_phi_ak/ak - (2.0*a_tt/(ak*a_t*a_t) + 1.0/(ak*ak))*lin_phi - (0.5/(Mpl*Mpl))*(lin_delf*V_fvl/(a_t*a_t) 
			+ lin_phi*fb_ak*fb_ak - fb_ak*lin_delf_ak) - a_tt*lin_phi_ak/(a_t*a_t);

		lin_delf_a = lin_delf_a + (lin_delfac1+lin_delfac2)*0.5*da;
       		lin_phi_a = lin_phi_a + (lin_phiac1+lin_phiac2)*0.5*da;
       



	}


	


         
    }
    
    a_t = sqrt((ommi*ai*ai*ai/a  + (1.0/(Mpl*Mpl))*a*a*Vvl/(3.0*c)) / ( 1.0 - (1.0/(Mpl*Mpl))*a*a*fb_a*fb_a/(6.0*c*c*c))) ;
    Hi = Hb0*a/a_t;
    printf("\nHi    %.20lf  \nratio(Hi/Hb0)  %.20lf\n",Hi,a/a_t);
    Vvl = V(fb);
    
    w = (fb_a*fb_a*a_t*a_t/(2.0*c*c) - Vvl)/(fb_a*fb_a*a_t*a_t/(2.0*c*c) + Vvl);

  

 	fprintf(fpback,"\n\n\n");


   
    
   
 
    


    

  
}


void write_fields()
{
	int i;
	char name_p[20],name_f[20];
	double f_dc,f_prsr, f_denst, Vvl, Vvlb,back_f_denst,zaw;

	Vvlb = V(fb);

	zaw = a0/a - 1.0;

	back_f_denst = (0.5*fb_a*a_t*fb_a*a_t + Vvlb);

	
	snprintf(name_f,20,"lin_fields_z_%lf",zaw);

	
	fp_fields = fopen(name_f,"w");

	for(i=0;i<tN;++i)
	{
		Vvl = V(f[i]);	
				

		f_prsr = 0.5*( f_a[i]*a_t*f_a[i]*a_t/(1.0+2.0*(phi[i]))
			 - (f_s[0][i]*f_s[0][i]+f_s[1][i]*f_s[1][i]+f_s[2][i]*f_s[2][i])/(a*a*(1.0-2.0*phi[i])) ) - Vvl;
		f_denst = 0.5*( f_a[i]*a_t*f_a[i]*a_t/(1.0+2.0*(phi[i]))
			 - (f_s[0][i]*f_s[0][i]+f_s[1][i]*f_s[1][i]+f_s[2][i]*f_s[2][i])/(a*a*(1.0-2.0*phi[i])) ) + Vvl;

		f_dc = (f_denst/back_f_denst)-1.0;

		fprintf(fp_fields,"%d\t%lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf\n",
					i,a/ai,grid[i][0],grid[i][1],grid[i][2],density_contrast[i],phi[i],0.0,f[i],f_dc,f_prsr/f_denst);

		
	}

	fclose(fp_fields);
	

}




void initialise()
{
      int l1,l2,r1,r2;
      double Vvlb;
    
      
      int xcntr[3]={-1,-1,-1},anchor[3],ci,j;
      double gamma, v, gradmagf;
      double ktmp,maxkmagsqr = 0.0;
      double wktmp,shtmp;
      a0 = 1.00;
      ai = 0.001;
      a = ai;
      omdmbini= (cpmc)*pow((a0/ai),3.0)/(cpmc*a0*a0*a0/(ai*ai*ai) + (1.0-cpmc));
      printf("omdmbini  %.10lf\n",omdmbini);

      a_t=Hi*ai;
	
	lin_i = 10;

	kf = tpie*lenfac/(64.0);



	dx[0] = 1.0; dx[1] =1.0; dx[2] = 1.0;
        L[0] = dx[0]*((double) (n));  L[1] = dx[1]*((double) (n));  L[2] = dx[2]*((double) (n));

		for(j=0;j<3;++j)
	{
		d1[j] = 12.0*dx[j];
		d2[j] = 12.0*dx[j]*dx[j];

	} 
	

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
	background(1);

	Vamp = Vamp*Hi*Hi; 


	a = a_zels;

	fb = fb_zeldo;
	fb_a = fb_a_zeldo;

	Vvlb = V(fb_zeldo);

	a_t = sqrt((Hi*Hi*omdmbini*ai*ai*ai/a_zels  + 
			(1.0/(Mpl*Mpl))*a_zels*a_zels*Vvlb/(3.0*c)) / ( 1.0 - (1.0/(Mpl*Mpl))*a_zels*a_zels*fb_a*fb_a/(6.0*c*c*c))) ;
       a_tt = -0.5*omdmbini*Hi*Hi*ai*ai*ai/(a_zels*a_zels) - (1.0/(Mpl*Mpl*c))*a_zels*(fb_a*fb_a*a_t*a_t - Vvlb)/3.0;


	 fbdss =  (-0.5*fb_a*fb_a*a_t*a_t + Vvlb) ;
	 fb00 =  (0.5*fb_a*fb_a*a_t*a_t + Vvlb) ;

//#################################### Linear Theory initialization #################################

	lin_phi = lin_phi_zeldo;
   	lin_delf = lin_delf_zeldo;
   	lin_phi_a =  lin_phi_a_zeldo;
   	lin_delf_a = lin_delf_a_zeldo;
 
//####################################################################################################


	for(ci=0;ci<tN;++ci)
	  {

		f[ci] = fb_zeldo;
		f_a[ci] = fb_a_zeldo;
	
		
		

		tul00[ci]= 0.0;
		tuldss[ci]=0.0;
		
			
			

		/*	if(ci==1)
		{printf("\n anchor %lf\t%lf\t%lf\n\n",grid[anchor[0]][0],grid[anchor[1]][1],grid[anchor[2]][2]);

			for(j=0;j<8;++j)
			{
				printf("\n anchor %lf\t%lf\t%lf",grid[p[ci].cubeind[j]][0],grid[p[ci].cubeind[j]][0],grid[p[ci].cubeind[j]][0]);

 			}
		
		// printf("\n anchor %lf\t%lf\t%lf\n\n",p[ci].x[0],p[ci].x[1],p[ci].x[2]);
			printf("\n\n");		 
		}
		*/

	}


	
	           
          
	
	cal_grd_tmunu(0);

	cal_dc();
	
	

	

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

	printf("\n Linear theory kf is %lf kl %lf Mpc %lf  %lf\n",kf,tpie*lenfac/kf,lin_phi,lin_delf);
	

	  

}



void cal_grd_tmunu(int k)
{
	int ci,l1,l2,r1,r2,j,Vvl,V_fvl,fl;

	


 	if(k==1)
	{	fftw_execute(scf_plan_b);
	#pragma omp parallel for private(j,l1,l2,r1,r2,Vvl,V_fvl)
	  for(ci=0;ci<tN;++ci)
	   {
	  //  particle2mesh(tmpp,ci,tmpphi,a);

	    
	    

	   
	   // LAPphi[ci] = 0.0;
	    LAPf[ci] = 0.0;
	  
	    
	     for(j=0;j<3;++j)
	     {	 
		


		l1 = ci + ((n+ind_grid[ci][j]-1)%n)*((int)(pow(n,2-j))) - ind_grid[ci][j]*((int)(pow(n,2-j)));
		l2 = ci + ((n+ind_grid[ci][j]-2)%n)*((int)(pow(n,2-j))) - ind_grid[ci][j]*((int)(pow(n,2-j)));
		r1 = ci + ((n+ind_grid[ci][j]+1)%n)*((int)(pow(n,2-j))) - ind_grid[ci][j]*((int)(pow(n,2-j)));
		r2 = ci + ((n+ind_grid[ci][j]+2)%n)*((int)(pow(n,2-j))) - ind_grid[ci][j]*((int)(pow(n,2-j)));
		
		
		phi_s[j][ci] = (phi[l2]-8.0*phi[l1]+8.0*phi[r1]-phi[r2])/(d1[j]);
		//f_s[j][ci] = (tmpf[l2]-8.0*tmpf[l1]+8.0*tmpf[r1]-tmpf[r2])/(12.0*dx[j]); 
		
		
		//LAPphi[ci] += (-tmpphi[l2]+16.0*tmpphi[l1]-30.0*tmpphi[ci]+16.0*tmpphi[r1]-tmpphi[r2])/(12.0*dx[j]*dx[j]);
		LAPf[ci] += (-scf_rhs[l2][0]+16.0*scf_rhs[l1][0]-30.0*scf_rhs[ci][0]+16.0*scf_rhs[r1][0]-scf_rhs[r2][0])/(n3sqrt*d2[j]); 

		
		

		
		
	     }
  		

		f[ci] = scf_rhs[ci][0]/n3sqrt;
		
		Vvl = V(f[ci]);


		
		
		
		tuldss[ci]+=3.0*(Vvl - 0.5*tmpf_a[ci]*tmpf_a[ci]*a_t*a_t*(1.0-2.0*phi[ci]) - fbdss);



	  }
	}




	else
	{
	 #pragma omp parallel for private(j,l1,l2,r1,r2,Vvl,V_fvl,fl)
	  for(ci=0;ci<tN;++ci)
	   {
	   //particle2mesh(p,ci,phi,a);

	    

	  
	

		V_fvl = V_f(f[ci]);
		Vvl = V(f[ci]);



		
	
		


		fl = ( V_fvl/(a_t*a_t) + 3.0*f_a[ci]/a - 3.0*f_a[ci]*phi_a[ci] - 6.0*(phi[ci])*f_a[ci]/a 
				- (phi_a[ci])*f_a[ci]
			)/(-1.0+2.0*(phi[ci]))
			-a_tt*f_a[ci]/(a_t*a_t)  -2.0*(LAPf[ci]/a)*(2.0*phi[ci])/(a*a_t*a_t)  ; 

		scf_rhs[ci][0] = f[ci] + da*f_a[ci]*a_t + 0.5*da*da*fl;	
		scf_rhs[ci][1] = 0.0;



		
		tul00[ci]+= (Vvl + 0.5*f_a[ci]*f_a[ci]*a_t*a_t*(1.0-2.0*phi[ci])) -fb00 ;
		tuldss[ci]+= 3.0*((Vvl - 0.5*f_a[ci]*f_a[ci]*a_t*a_t*(1.0-2.0*phi[ci])) -fbdss) ;
  		


	  }
	fftw_execute(scf_plan_f);


	}



}







int evolve(double aini, double astp)
{
    printf("Evvvolove\n");


    double ommi = omdmbini;
    double facb1,facb2,Vvl,V_fvl,fb_ak,fbk,omfb,Vvlb,V_fvlb,V_ffvlb,lin_delfac1,lin_delfac2,lin_phiac1,lin_phiac2,lin_delf_ak,lin_phi_ak,kfac2;
    double w;

    int i,j,lcntr,ci;
     
     ///Watch out for local vs global for parallelization
    double phiacc1[n*n*n],phiacc2[n*n*n],facc1[n*n*n],facc2[n*n*n];



    
  
   

  for(a=aini,lcntr=0;((a/ai)<=astp)&&(fail==1);++lcntr)
    { //if(lcntr%jprint==0)
	   
          
      
      Vvlb = V(fb);
      V_fvlb =  V_f(fb);
      V_ffvlb =  V_ff(fb);

      a_t = sqrt((Hi*Hi*ommi*ai*ai*ai/a  + (1.0/(Mpl*Mpl))*a*a*Vvlb/(3.0*c)) / ( 1.0 - (1.0/(Mpl*Mpl))*a*a*fb_a*fb_a/(6.0*c*c*c))) ;
      a_tt = -0.5*ommi*Hi*Hi*ai*ai*ai/(a*a) - (1.0/(Mpl*Mpl*c))*a*(fb_a*fb_a*a_t*a_t - Vvlb)/3.0;


      facb1 = -V_fvlb*c*c/(a_t*a_t) - 3.0*fb_a/a - a_tt*fb_a/(a_t*a_t);

	
	omfb = (1.0/(3.0*c*c*c*Mpl*Mpl))*(fb_a*fb_a*a_t*a_t/(2.0*c*c) + Vvlb);


	lin_dcf = (V_fvlb*lin_delf - lin_phi*fb_a*a_t*fb_a*a_t + fb_a*a_t*lin_delf_a*a_t)/(0.5*fb_a*a_t*fb_a*a_t + Vvlb);
	lin_dcm = -(2.0*a*a*a/(3.0*Hi*Hi*ommi*ai*ai*ai))*( 3.0*lin_phi*a_t*a_t/(a*a) + 3.0*lin_phi_a*a_t*a_t/(a) + kf*kf*lin_phi/(a*a) )
											 - omfb*lin_dcf*a*a*a/(ommi*Hi*Hi*ai*ai*ai) ;

	lin_growth = lin_dcm*lin_ini_phi/lin_ini_dcm;


      lin_delfac1 =  -3.0*lin_delf_a/a - kf*kf*lin_delf/(a*a*a_t*a_t) - 2.0*lin_phi*V_fvlb/(a_t*a_t) + 4.0*lin_phi_a*fb_a- V_ffvlb*lin_delf/(a_t*a_t) 
				- a_tt*lin_delf_a/(a_t*a_t);

	

	
	lin_phiac1 = -4.0*lin_phi_a/a - (2.0*a_tt/(a*a_t*a_t) + 1.0/(a*a))*lin_phi - (0.5/(Mpl*Mpl))*(lin_delf*V_fvlb/(a_t*a_t) 
			+ lin_phi*fb_a*fb_a - fb_a*lin_delf_a) - a_tt*lin_phi_a/(a_t*a_t);

	

      fb_ak = fb_a + facb1*da;
      fb = fb + fb_a*da+0.5*facb1*da*da; 
      lin_delf = lin_delf + lin_delf_a*da + 0.5*lin_delfac1*da*da;
      lin_delf_ak = lin_delf_a + lin_delfac1*da;
      lin_phi = lin_phi + lin_phi_a*da + 0.5*lin_phiac1*da*da;
      lin_phi_ak = lin_phi_a + lin_phiac1*da;
         
	  if(lcntr%jprint==0)
	  { 
		 omfb = (1.0/(3.0*c*c*c*Mpl*Mpl))*(fb_a*fb_a*a_t*a_t/(2.0*c*c) + Vvlb);

		fprintf(fpback,"%lf\t%.10lf\t%.10lf\n",a/ai,ommi*ai*ai*ai*Hi*Hi/(a*a_t*a_t),omfb*a*a/(a_t*a_t));
		fprintf(fplin,"%lf\t%.20lf\t%.20lf\n",a/ai,lin_growth,a/ai);

		printf("a  %lf %.10lf  %.10lf\n",a,ommi*ai*ai*ai*Hi*Hi/(a*a_t*a_t),omfb);

	
		}

	
	if((lcntr%jprints==0))
	   { printf("printing..\n");

		 cal_dc();
      		 cal_spectrum(density_contrast,fppwspctrm_dc,0);
		// cal_spectrum(phi,fppwspctrm_phi,0);
		 write_fields();


	  }

/////////////////////////////////particle force calculation*****Step 1////////////////////////////////////////////////		 
	#pragma omp parallel for private(i,kfac2)
	 for(ci=0;ci<tN;++ci)
	  {
		
/////////////////////phi acceleration calculation Step 1/////////////////////////////////////////////////////////////////////////////////

		phiacc1[ci] = (1.0/(a_t*a*a_t*a))*(- 2.0*a*phi[ci]*a_tt 
					      -a*a*tuldss[ci]/(6.0*Mpl*Mpl))  -phi[ci]/(a*a) 
						- 3.0*phi_a[ci]/a -phi_a[ci]/a - a_tt*phi_a[ci]/(a_t*a_t);

		V_fvl = V_f(f[ci]);
		Vvl = V(f[ci]);
	
		facc1[ci] = ( V_fvl/(a_t*a_t) + 3.0*f_a[ci]/a - 3.0*f_a[ci]*phi_a[ci] - 6.0*phi[ci]*f_a[ci]/a 
				- phi_a[ci]*f_a[ci] -LAPf[ci]*(1.0+2.0*phi[ci])/(a*a*a_t*a_t) )/(-1.0+2.0*phi[ci])-a_tt*f_a[ci]/(a_t*a_t); 

		
		

		//phiacc = (1.0/(2.0*a_t*a*a_t*a))*(-2.0*phi[ci]*a_t*a_t - 4.0*a*phi[ci]*a_tt 
			//		      -a*a*tuldss[ci]/(3.0*Mpl*Mpl)) - 3.0*phi_a[ci]/a -phi_a[ci]/a - a_tt*phi_a[ci]/(a_t*a_t);
		


		kfac2 = tpie*tpie*(k_grid[ci][0]*k_grid[ci][0]+k_grid[ci][1]*k_grid[ci][1]+k_grid[ci][2]*k_grid[ci][2]);
	    
	        scf_rhs_ft[ci][0] = (scf_rhs_ft[ci][0]/n3sqrt)/(1.0+0.5*kfac2*(da/(a_t*a))*(da/(a_t*a))); 
	        scf_rhs_ft[ci][1] = (scf_rhs_ft[ci][1]/n3sqrt)/(1.0+0.5*kfac2*(da/(a_t*a))*(da/(a_t*a)));



		
		phi[ci]  = phi[ci]+da*phi_a[ci]+0.5*da*da*phiacc1[ci];
		tmpphi_a[ci] = phi_a[ci]+da*phiacc1[ci];

		
		tmpf_a[ci] = f_a[ci]+da*facc1[ci];


		if(isnan(phi[ci]+f[ci]))
				fail=0;

		

		
		tul00[ci] = 0.0 ;
		tuldss[ci] = 0.0;
		
		
	  }

		//fprintf(fp_particles,"\n\n\n");
		
	 



		
	  ak = a + da;

	  Vvlb = V(fb);
          V_fvlb =  V_f(fb);
	  V_ffvlb =  V_ff(fb);

	  a_t = sqrt((ommi*Hi*Hi*ai*ai*ai/ak  + (1.0/(Mpl*Mpl))*ak*ak*Vvlb/(3.0*c)) / ( 1.0 - (1.0/(Mpl*Mpl))*ak*ak*fb_ak*fb_ak/(6.0*c*c*c))) ;
          a_tt = -0.5*ommi*Hi*Hi*ai*ai*ai/(ak*ak) - (1.0/(Mpl*Mpl*c))*ak*(fb_ak*fb_ak*a_t*a_t - Vvlb)/3.0;


	  fbdss =  (-0.5*fb_ak*fb_ak*a_t*a_t + Vvlb) ;
	  fb00 =  (0.5*fb_ak*fb_ak*a_t*a_t + Vvlb) ;
 
	   
/////////////////////Intermediate Tul calculations and psi construction//////////////////////////////////////////


	cal_grd_tmunu(1);
	
	



		
 
	




/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 

	


       facb2 = -V_fvlb*c*c/(a_t*a_t) - 3.0*fb_ak/ak - a_tt*fb_ak/(a_t*a_t);

	

       
	lin_delfac2 =  -3.0*lin_delf_ak/ak - kf*kf*lin_delf/(ak*ak*a_t*a_t) - 2.0*lin_phi*V_fvlb/(a_t*a_t) + 4.0*lin_phi_ak*fb_ak- V_ffvlb*lin_delf/(a_t*a_t) 
				- a_tt*lin_delf_ak/(a_t*a_t);

	

	
	lin_phiac2 = -4.0*lin_phi_ak/ak - (2.0*a_tt/(ak*a_t*a_t) + 1.0/(ak*ak))*lin_phi - (0.5/(Mpl*Mpl))*(lin_delf*V_fvlb/(a_t*a_t) 
			+ lin_phi*fb_ak*fb_ak - fb_ak*lin_delf_ak) - a_tt*lin_phi_ak/(a_t*a_t);

	

       fb_a = fb_a + (facb1+facb2)*0.5*da;
       lin_delf_a = lin_delf_a + (lin_delfac1+lin_delfac2)*0.5*da;
       lin_phi_a = lin_phi_a + (lin_phiac1+lin_phiac2)*0.5*da;
       

	if(isnan(fb_a+fb))
	{	fail=0;//printf("%d Alert %lf\n",lcntr,facb2,fb_a);
		//break;
		
	}



////////////////////////////Final Step//////////////////////////////////////////////////////////////////////////////////////
          
 	  
	  

	 
	#pragma omp parallel for private(i)
	 for(ci=0;ci<tN;++ci)
	  {
		

			
/////////////////////phi  acceleration calculation Final/////////////////////////////////////////////////////////////////////////////////
		
		phiacc2[ci] = (1.0/(a_t*ak*a_t*ak))*(- 2.0*ak*phi[ci]*a_tt 
					      -ak*ak*tuldss[ci]/(6.0*Mpl*Mpl))  -phi[ci]/(ak*ak) 
						- 3.0*tmpphi_a[ci]/ak -tmpphi_a[ci]/ak - a_tt*tmpphi_a[ci]/(a_t*a_t);


		V_fvl = V_f(f[ci]);
		Vvl = V(f[ci]);
	
		facc2[ci] = ( V_fvl/(a_t*a_t) + 3.0*tmpf_a[ci]/a - 3.0*tmpf_a[ci]*phi_a[ci] - 
				6.0*phi[ci]*tmpf_a[ci]/a - tmpphi_a[ci]*tmpf_a[ci] 
				-LAPf[ci]*(1.0+2.0*phi[ci])/(a*a*a_t*a_t) )/(-1.0+2.0*phi[ci])-a_tt*tmpf_a[ci]/(a_t*a_t); 


		
		//phiacc = (1.0/(2.0*a_t*ak*a_t*ak))*(-2.0*tmpphi[ci]*a_t*a_t - 4.0*ak*tmpphi[ci]*a_tt 
		//		-ak*ak*tuldss[ci]/(3.0*Mpl*Mpl)) - 3.0*tmpphi_a[ci]/ak -tmpphi_a[ci]/ak - a_tt*tmpphi_a[ci]/(a_t*a_t);
		

		
		
		phi_a[ci] = phi_a[ci]+0.5*da*(phiacc1[ci]+phiacc2[ci]);
		f_a[ci] = f_a[ci]+0.5*da*(facc1[ci]+facc2[ci]);

 		if(isnan(phi[ci]+phi_a[ci]))
		fail=0;
		
		tul00[ci] = 0.0 ;
		tuldss[ci] = 0.0;


	}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////Final Tul and phi recosntruction/////////////////////////////////////////////////////////////
	 a = a+da;		


      a_t = sqrt((Hi*Hi*ommi*ai*ai*ai/a  + (1.0/(Mpl*Mpl))*a*a*Vvlb/(3.0*c)) / ( 1.0 - (1.0/(Mpl*Mpl))*a*a*fb_a*fb_a/(6.0*c*c*c))) ;
      a_tt = -0.5*ommi*Hi*Hi*ai*ai*ai/(a*a) - (1.0/(Mpl*Mpl*c))*a*(fb_a*fb_a*a_t*a_t - Vvlb)/3.0;

	fbdss =  (-0.5*fb_a*fb_a*a_t*a_t + Vvlb) ;
	fb00 =  (0.5*fb_a*fb_a*a_t*a_t + Vvlb) ;

	


	

	cal_grd_tmunu(0);
	
		
	


	//fprintf(fp_particles,"\n\n\n");
	
	

     

 //   printf("evolve w  %.10lf  Hi %.10lf  %.10lf  %.10lf\n",a_t,a,a0);

    if(fail!=1)
    {//printf("fail  %d %d  %lf\n",fail,lcntr,a); 
	return(fail);
    }    
	
  }
 return(fail);
}






