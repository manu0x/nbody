#include <fftw3.h>
#include <fftw3-mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h> 
#include <omp.h>
#include <mpi.h>
#include <fenv.h>
#include <time.h>
//#include "mt19937ar.c"

#define  n 64

#define tpie  2.0*M_PI

FILE *fppwspctrm;


/////////// MPI related ////////////////////


int mpicheck = 0;
int num_p = 0;
int rank;


int nd_cart;
int *my_coords;
int my_corank;
int n_axis_loc[3];

int * dims,*periods;

MPI_Datatype c_x_plain,c_y_plain,c_z_plain;
MPI_Status stdn,stup;


MPI_Comm cart_comm;


//////////////////////////////////////////////

double G   = 1.0;
double c   = 1.0;
double Mpl ;
double lenfac = 1.0;
double Hb0  ;
double L[3];
int tN;
int n_axis[3];
int fail =1;

clock_t t_start,t_end;

double n3sqrt;
double *phi, *phi_a,  *f,*f_a,*slip,*slip_a,*tul00,*tuldss,fbdss,fb00;
double *phi_s[3],*f_s[3],*slip_s[3],*LAPslip,*LAPf,*tmpslip2,*tmpslip1;
double *tmpphi,  *tmpf,*tmpphi_a, *tmpf_a, *ini_vel0,*ini_vel1,*ini_vel2,m=1.0;
double dx[3],d1[3],d2[3];
double *density_contrast,*ini_density_contrast,*ini_phi_potn;
double *scf_holder;

/*struct particle
	{	
		double x[3];
		double v[3];
		
		
		int 	cubeind[8];	

	};



struct particle p[n*n*n],tmpp[n*n*n];
*/
double *grid[3];
int *ind_grid[3];
double *k_grid[3];
int *kmagrid,kbins,*kbincnt;
double dk; double *pwspctrm;
double *W_cic,*C1_cic_shot;

 


 
  
  

fftw_complex *ini_del;
fftw_complex *F_ini_del;
fftw_complex *F_ini_phi;
fftw_complex *ini_phi;



fftw_plan ini_del_plan;
fftw_plan ini_phi_plan;

fftw_complex *slip_rhs;
fftw_complex *slip_rhs_ft;



fftw_plan slip_plan_f;
fftw_plan slip_plan_b;


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
void ini_rand_field();
void read_ini_rand_field();

int evolve(double ,double );
void cal_spectrum(double *,FILE *,int);
void cal_dc();
void write_fields();
void slip_fft_cal();
double V(double);
double V_f(double);
double V_ff(double);
void cal_grd_tmunu();


void allocate_fields(int *);
void allocate_fft_fields(int *);


void main(int argc, char **argv)
{   t_start = clock();


	int i;
      

	n_axis[0]=n;
	n_axis[1]=n;
	n_axis[2]=n;


	mpicheck = MPI_Init(&argc,&argv);
	mpicheck = MPI_Comm_size(MPI_COMM_WORLD,&num_p);
	mpicheck = MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	
	
	fftw_mpi_init();
	

	if(num_p<4)
		nd_cart = 1;
	else
	{	if(num_p<20)
			nd_cart = 2;
		else
			nd_cart = 3;
	}

	 dims = calloc(nd_cart,sizeof(int));
	 periods = calloc(nd_cart,sizeof(int));
	my_coords = calloc(nd_cart,sizeof(int));

	for(i=0;i<nd_cart;++i)
	{	periods[i] = 1;
		


	}


	mpicheck = MPI_Dims_create(num_p,nd_cart,dims);

	mpicheck = MPI_Cart_create(MPI_COMM_WORLD,nd_cart, dims, periods,1,&cart_comm);

	mpicheck = MPI_Cart_get(cart_comm,nd_cart,dims,periods,my_coords);
	
	mpicheck = MPI_Cart_rank(cart_comm,my_coords,&my_corank);

	for(i=0;i<3;++i)
	{	
		if(i<nd_cart)
		{
		   double temp_naxis = (((double) n_axis[i])/ ((double) dims[i]));
		   n_axis_loc[i] = (int) temp_naxis+1;


		    if((n_axis[i]%dims[i]) != 0)
		     {
		     
			 if(my_coords[i]==(dims[i]-1))
		      		n_axis_loc[i]+=(n_axis[i]-dims[i]*n_axis_loc[i]);

		      }

		}

		else
			n_axis_loc[i] = n_axis[i]; 


	}

	
	
  MPI_Type_vector(n_axis_loc[1],n_axis_loc[2],n_axis_loc[2]+4,MPI_DOUBLE,&c_x_plain);
  MPI_Type_commit(&c_x_plain);
  
  MPI_Type_vector(n_axis_loc[0],n_axis_loc[2],(n_axis_loc[2]+4)*(n_axis_loc[1]+4),MPI_DOUBLE,&c_y_plain);
  MPI_Type_commit(&c_y_plain);
  
  MPI_Type_vector(n_axis_loc[0]*n_axis_loc[1],1,n_axis_loc[2]+4,MPI_DOUBLE,&c_z_plain);
  MPI_Type_commit(&c_z_plain);


	
	

	



    Mpl = 1.0/sqrt(8.0*3.142*G) ;
	Hb0  = 22.04*(1e-5)*lenfac;

        da = 1e-5;
        jprint = (int) (0.001/da);
	jprints = 200*jprint;
	
	tN=(n_axis_loc[0]+2)*(n_axis_loc[1]+2)*(n_axis_loc[2]+2);

	n3sqrt = sqrt((double) tN);
        
	printf("jprint %d tN %d  Hb0 %.10lf\n",jprint,tN,Hb0); 
	//feenableexcept(FE_DIVBYZERO | FE_ItNVALID | FE_OVERFLOW);
	//feenableexcept(FE_DIVBYZERO | FE_ItNVALID | FE_OVERFLOW);

	//fftwl_init_threads();
	//fftwl_plan_with_nthreads(128);

	
	
	
	//fpdc  = fopen("dc.txt","w");
	fpback  = fopen("back.txt","w");
	/*fppwspctrm_dc  = fopen("pwspctrm_dc2.txt","w");
	fppwspctrm_phi  = fopen("pwspctrm_phi.txt","w");
	fpphi = fopen("phi.txt","w");
	
	fplinscale = fopen("linscale.txt","w");
	fplin = fopen("lpt.txt","w");
	*/

   

       // i = fftw_init_threads();
	//	fftw_plan_with_nthreads(omp_get_max_threads());

	

	allocate_fields(n_axis_loc);
	allocate_fft_fields(n_axis_loc);

       
	

	//m = (double *) malloc(n*n*n*sizeof(double)); 
	
	
      
	
	background(0);
	initialise();
	
/*
       i = evolve(a_zels,a0/ai);
	
       cal_dc();
       cal_spectrum(density_contrast,fppwspctrm_dc,0);
	write_fields();
	
	if(i!=1)
	printf("\nIt's gone...\n");

	t_end = clock();

	printf("\nTotal consumed time is %lf\n",(double) ((t_end-t_start)/CLOCKS_PER_SEC));


*/
fftw_mpi_cleanup();

MPI_Finalize();

}







void allocate_fields(int nax[3])
{

	 	
	
	int l = (nax[0]+4)*(nax[1]+4)*(nax[2]+4);
	
	phi  = calloc(l,sizeof(double));
	phi_a  = calloc(l,sizeof(double));
	f  = calloc(l,sizeof(double));
	scf_holder  = calloc(l,sizeof(double));
	f_a  = calloc(l,sizeof(double));
	slip  = calloc(l,sizeof(double));
	slip_a  = calloc(l,sizeof(double));
	tul00  = calloc(l,sizeof(double));
	tuldss  = calloc(l,sizeof(double));

	phi_s[0]  = calloc(l,sizeof(double));
	phi_s[1]  = calloc(l,sizeof(double));
	phi_s[2]  = calloc(l,sizeof(double));
	
	f_s[0]  = calloc(l,sizeof(double));
	f_s[1]  = calloc(l,sizeof(double));
	f_s[2]  = calloc(l,sizeof(double));
	
	
	slip_s[0]  = calloc(l,sizeof(double));
	slip_s[1]  = calloc(l,sizeof(double));
	slip_s[2]  = calloc(l,sizeof(double));



	

	LAPslip  = calloc(l,sizeof(double));
	LAPf = calloc(l,sizeof(double));
	tmpslip1  = calloc(l,sizeof(double));
	tmpslip2  = calloc(l,sizeof(double));

	density_contrast = calloc(l,sizeof(double));
	
	int i;
	
	for(i=0;i<3;++i)
	{
		ind_grid[i] = calloc(l,sizeof(int));
		grid[i] = calloc(l,sizeof(double));
		k_grid[i] = calloc(l,sizeof(double));
	
	}


	printf("Allocated...\n");


}




void allocate_fft_fields(int nax[3])
{

	 	
	 	
	 	
	
	int l = (nax[0]+2)*(nax[1]+2)*(nax[2]+2);
	
	const ptrdiff_t n0 = nax[0];
	const ptrdiff_t n1 = nax[1];
	const ptrdiff_t n2 = nax[2];
	
	
	fftw_complex *data;
    ptrdiff_t alloc_local, local_n0, local_0_start;
	
	
	alloc_local = fftw_mpi_local_size_3d(n0, n1, n2,
                                 cart_comm,
                                 &local_n0, &local_0_start);
	

	int a,b,c;
	a = nax[0]+2;
	b = nax[1]+2;
	c = nax[2]+2;
	

	
	//slip_rhs = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * l);
	//slip_rhs_ft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * l);
	
	slip_rhs = fftw_alloc_complex(alloc_local);
	slip_rhs_ft = fftw_alloc_complex(alloc_local);


	//scf_rhs = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * l);
	//scf_rhs_ft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * l);
	
	
	scf_rhs = fftw_alloc_complex(alloc_local);
	scf_rhs_ft = fftw_alloc_complex(alloc_local);
	
	
	
	

	//slip_plan_f = fftw_plan_dft_3d(a,b,c, slip_rhs, slip_rhs_ft, FFTW_FORWARD, FFTW_ESTIMATE);
	//slip_plan_b = fftw_plan_dft_3d(a,b,c, slip_rhs_ft, slip_rhs, FFTW_BACKWARD, FFTW_ESTIMATE);
	
	slip_plan_f = fftw_mpi_plan_dft_3d(n0, n1,  n2,
                               slip_rhs, slip_rhs_ft,
                              cart_comm, FFTW_FORWARD, FFTW_ESTIMATE);
                              
    slip_plan_b =  fftw_mpi_plan_dft_3d(n0, n1,  n2,
                              slip_rhs_ft, slip_rhs,
                              cart_comm, FFTW_BACKWARD, FFTW_ESTIMATE);
	

	//scf_plan_f = fftw_plan_dft_3d(a,b,c, scf_rhs, scf_rhs_ft, FFTW_FORWARD, FFTW_ESTIMATE);
	//scf_plan_b = fftw_plan_dft_3d(a,b,c, scf_rhs_ft, scf_rhs, FFTW_BACKWARD, FFTW_ESTIMATE);
	
	scf_plan_f =  fftw_mpi_plan_dft_3d(n0, n1,  n2,
                               scf_rhs, scf_rhs_ft,
                              cart_comm, FFTW_FORWARD, FFTW_ESTIMATE);
                              
    scf_plan_b =  fftw_mpi_plan_dft_3d(n0, n1,  n2,
                              scf_rhs_ft, scf_rhs,
                              cart_comm, FFTW_BACKWARD, FFTW_ESTIMATE);
	
	

	printf("Allocated...fft_fields\n");


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
		if(i<=kbins)
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
	
		
		
		t00f_loc+=  0.5*(1.0+2.0*phi[i])*f_s[j][i]*f_s[j][i]/(a*a)  ;

		
		

		
		
	     }


		Vvl = V(f[i]);


		t00f_loc+=Vvl + 0.5*f_a[i]*f_a[i]*a_t*a_t*(1.0-2.0*(phi[i]-slip[i])) - fb00;

		density_contrast[i] = (-3.0*a_t*a_t*(phi[i]-slip[i])/(a*a) + lapphi_loc/(a*a) -3.0*a_t*phi_a[i]*a_t/a -0.5*t00f_loc/(Mpl*Mpl))/
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
{	//init_genrand(time(0));
	int i,j,k,ief,jef,kef,cnt,rcnt,rk,ri,rj,maxcnt=0; 
	double ksqr,muk,sigk;
	double a1,a2,b1,b2,a,b;

	FILE *fpinirand = fopen("initial_rand_field.txt","w");


	
	double zdvfac = -(2.0/3.0)*a_t/(cpmc*Hb0*Hb0);


	//init_genrand(time(0));
	
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
		 	   // a1 = genrand_res53();
 			    //a2 = genrand_res53(); 
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
   		lin_delf_a = 0.0;///ini_phi_potn[lin_i];
 



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

	
	snprintf(name_f,20,"fields_z_%lf",zaw);


	fp_fields = fopen(name_f,"w");

	for(i=0;i<tN;++i)
	{
		Vvl = V(f[i]);	
				

		f_prsr = 0.5*( f_a[i]*a_t*f_a[i]*a_t/(1.0+2.0*(phi[i]-slip[i]))
			 - (f_s[0][i]*f_s[0][i]+f_s[1][i]*f_s[1][i]+f_s[2][i]*f_s[2][i])/(a*a*(1.0-2.0*phi[i])) ) - Vvl;
		f_denst = 0.5*( f_a[i]*a_t*f_a[i]*a_t/(1.0+2.0*(phi[i]-slip[i]))
			 - (f_s[0][i]*f_s[0][i]+f_s[1][i]*f_s[1][i]+f_s[2][i]*f_s[2][i])/(a*a*(1.0-2.0*phi[i])) ) + Vvl;

		f_dc = (f_denst/back_f_denst)-1.0;

		fprintf(fp_fields,"%d\t%lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf\t%.20lf\n",
					i,a/ai,grid[i][0],grid[i][1],grid[i][2],density_contrast[i],phi[i],slip[i],f[i],f_dc,f_prsr/f_denst);

		


	}

	fclose(fp_fields);
	
}




void initialise()
{
      int l1,l2,r1,r2,bog_i;
      double Vvlb;

     
     //FILE *fpinirand = fopen("initial_rand_field.txt","r");
     
      int px,py,pz,ci[3],pgi,j,loc_ci[3],cci;
      int xcntr[3]={-1,-1,-1},anchor[3];
      double gamma, v, gradmagf;
      double ktmp,maxkmagsqr = 0.0,minkmagsqr = 1e10;
      double wktmp,shtmp;
      
      double ini_density_contrast_read,ini_phi_potn_read;
      
      int tmp_naxis[3];
	int tmp_naxistart[3];
	int tmp_naxisend[3];
      int myloc_count[3];
      a0 = 1.00;
      ai = 0.001;
      a = ai;
      omdmbini= (cpmc)*pow((a0/ai),3.0)/(cpmc*a0*a0*a0/(ai*ai*ai) + (1.0-cpmc));
      printf("omdmbini  %.10lf  ndcart %d\n",omdmbini,nd_cart);

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
	
	


	for(j=0;j<nd_cart;++j)
	{
		tmp_naxis[j] = (int)(((double) n_axis[j])/ ((double) dims[j]));

		tmp_naxistart[j] = my_coords[j]*tmp_naxis[j];

		tmp_naxisend[j] = tmp_naxistart[j]+n_axis_loc[j];

		myloc_count[j] = 0;
		
		printf("rank %d i %d start %d end %d\n",my_corank,j,tmp_naxistart[j],tmp_naxisend[j]);

	

	}


	if(nd_cart<3)
	{	

		if(nd_cart<2)
		{	
			tmp_naxistart[1] = 0;

			//tmp_naxisend[1] = tmp_naxistart[1]+n_axis_loc[1];
		
		
		
		}
		
		
		tmp_naxistart[2] = 0;

		tmp_naxisend[2] = tmp_naxistart[2]+n_axis_loc[2];
			
		
	printf("CC %d\t%d\t%d\n",tmp_naxistart[0],tmp_naxistart[1],tmp_naxistart[2]);


	}

	

       // L[0] = dx[0]*((double) (n));  L[1] = dx[1]*((double) (n));  L[2] = dx[2]*((double) (n));
	//dk = 0.01/dx[0]; kbins = 0;
	
	//ini_rand_field();
	 // read_ini_rand_field();
        
	//for(ci[0] = 0;ci[0] <tN; ++ci[0])
	//{
	//	kbincnt[ci]=0;
	
	//}
	
	
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

	
	
	
	
	
////////////////////////////////////////////3 -loop starts//////////////////////////////////	
	for(ci[0] = 0,loc_ci[0] = 1;ci[0] <n; ++ci[0])
	{	//printf("Y0Y0\n");

		if( (ci[0]>=tmp_naxistart[0])&&
			(ci[0]<tmp_naxisend[0])    )
		++loc_ci[0];
		
	 for(ci[1] = 0,loc_ci[1] = 1;ci[1] <n; ++ci[1])
	 {
	   if( (ci[1]>=tmp_naxistart[1])&&
			(ci[1]<tmp_naxisend[1])    )
		++loc_ci[1];

	   for(ci[2] = 0,loc_ci[2] = 1;ci[2] <n; ++ci[2])
	   {
		if( (ci[2]>=tmp_naxistart[2])&&
			(ci[2]<tmp_naxisend[2])    )
		++loc_ci[2];
		 
		
		//  fscanf(fpinirand,"%d\t%lf\t%lf\n",
		//			&bog_i,&ini_density_contrast_read,&ini_phi_potn_read);
  
		
		if(  (ci[0]>=tmp_naxistart[0])&&(ci[1]>=tmp_naxistart[1])&&(ci[2]>=tmp_naxistart[2]) &&

			(ci[0]<tmp_naxisend[0])&&(ci[1]<tmp_naxisend[1])&&(ci[2]<tmp_naxisend[2])     )
		{ 
			
			cci = loc_ci[0]*(n_axis_loc[1]+4)*(n_axis_loc[2]+4) + loc_ci[1]*(n_axis_loc[2]+4) + loc_ci[2];
	          for(j=0;j<3;++j)
	           {	//
					grid[j][cci] = ((double)(ci[j]))*dx[j];

					ind_grid[j][cci] = ci[j];
						
					if(ci[j]<=(n/2))
					{
					
					  k_grid[j][cci] = ( (double) ci[j])/L[j];

				  

					}
				else
				 { 
				 	 k_grid[j][cci] = ((double)(ci[j]-n))/L[j];

				
				  
				  }
		


		  }
		
		
		


		//W_cic[loc_ci] = wktmp*wktmp;
		//C1_cic_shot[loc_ci] =  shtmp;	
		
	
			
	/*	if(ktmp>maxkmagsqr)
		maxkmagsqr = (ktmp);
		if((ktmp>0.0)&&(minkmagsqr>ktmp))
		minkmagsqr = ktmp;
		

		kmagrid[ci] = (int)(sqrt(ktmp)/(dk));
		 //printf("yo  %d  %lf\n",kmagrid[ci],sqrt(ktmp));
		++kbincnt[kmagrid[ci]];

		if(kmagrid[ci]>kbins)
		kbins=kmagrid[ci];
		
	*/		
		f[cci] = fb_zeldo;
		f_a[cci] = fb_a_zeldo;
	
	     
		

		tul00[cci]= 0.0;
		tuldss[cci]=0.0;
	

		 phi[cci] =  ini_phi_potn_read;
		 phi_a[cci] = 0.0;
		 slip[cci] = 0.0;
		 slip_a[cci] = 0.0;
		 tmpslip2[cci] = 0.0;
		 tmpslip1[cci] = 0.0;
		 slip_s[0][cci] = 0.0;
		 slip_s[1][cci] = 0.0;
		 slip_s[2][cci] = 0.0;
		}
	    }
	  }
		
		
   }
   
  
////////////////////////////////////////////////////////////////3-loop ends//////////////////////
	printf("\nCCI is %d\n",cci);

	 cal_grd_tmunu();
	//cal_spectrum(ini_density_contrast,fppwspctrm_dc,1);

  


/*	for(ci=0;ci<tN;++ci)
	  {

		f[ci] = fb_zeldo;
		f_a[ci] = fb_a_zeldo;
	
	     
		

		tul00[ci]= 0.0;
		tuldss[ci]=0.0;
		
			
			


	}

	    
	cal_grd_tmunu();
*/	     
          

	
	
	  

}




void slip_fft_cal()
{    
	 double kfac,kfac2,tmp,tmptmp,Vvl,V_fvl,wl;
		
	 int i,l1,l2,r1,r2,j,mm,ci[3],cci,ccif,strt,strt_l,r_strt,r_strt_l,up,down;



	fftw_execute(scf_plan_b);

	
	


	 #pragma omp parallel for private(j,l1,l2,r1,r2,Vvl,V_fvl)
	 
	  for(ci[0]=2;ci[0]<(n_axis_loc[0]-2);++ci[0])
	  {
			
		for(ci[1]=2;ci[1]<(n_axis_loc[1]-2);++ci[1])
	  	{
		   for(ci[2]=2;ci[2]<(n_axis_loc[2]-2);++ci[2])
	  	  {
	  	   
			cci = ci[0]*(n_axis_loc[1]+4)*(n_axis_loc[2]+4) + ci[1]*(n_axis_loc[2]+4) + ci[2]+2;
			ccif = ci[0]*(n_axis_loc[1])*(n_axis_loc[2]) + ci[1]*(n_axis_loc[2]) + ci[2];
			
			f[cci] = scf_rhs[ccif][0]/n3sqrt;
	
	
		  }
		 }
	   }	  
	 ///////////////// make f data communication /////
	 
 strt_l = 2*(n_axis_loc[2]+4)*(n_axis_loc[1]+4)+2*(n_axis_loc[2]+4) + 2;
 strt = strt_l + (n_axis_loc[2]+4)*(n_axis_loc[1]+4)*(n_axis_loc[0]-1); 
 
 r_strt_l = 2*(n_axis_loc[2]+4) + 2;
 r_strt = strt + (n_axis_loc[2]+4)*(n_axis_loc[1]+4); 

 MPI_Cart_shift(cart_comm,0,1,&down,&up);
 
 MPI_Send((f+strt),1,c_x_plain,up,01,cart_comm);
 MPI_Recv((f+r_strt_l),1,c_x_plain,down,01,cart_comm,&stdn);
 
 MPI_Send((f+strt_l),1,c_x_plain,down,00,cart_comm);
 MPI_Recv(f+r_strt,1,c_x_plain,up,00,cart_comm,&stup);
 
 

	 
	 
	 
	 
	 
	 
	 /////////////////////////////////////////////////
	
	 for(ci[0]=2;ci[0]<(n_axis_loc[0]-2);++ci[0])
	  {
			
		for(ci[1]=2;ci[1]<(n_axis_loc[1]-2);++ci[1])
	  	{
		   for(ci[2]=2;ci[2]<(n_axis_loc[2]-2);++ci[2])
	  	  {
	  	   
			cci = ci[0]*(n_axis_loc[1]+4)*(n_axis_loc[2]+4) + ci[1]*(n_axis_loc[2]+4) + ci[2]+2;
			ccif = ci[0]*(n_axis_loc[1])*(n_axis_loc[2]) + ci[1]*(n_axis_loc[2]) + ci[2];
	

	    	
	   
	  
	  	  LAPf[cci] = 0.0;
	  	  wl = 1;
	   
	  for(j=2;j>-1;--j)
	  {	 
		

			
		l1 = cci - wl*(ci[j]) +  wl*(ci[j]-1) ;
		l2 = cci - wl*(ci[j]) +  wl*(ci[j]-2) ;
		r1 = cci - wl*(ci[j]) +  wl*(ci[j]+1) ;
		r2 = cci - wl*(ci[j]) +  wl*(ci[j]+2) ;


		wl*=(n_axis_loc[j]+4);
		
		
		f_s[j][cci] = (f[l2]-8.0*f[l1]+8.0*f[r1]-f[r2])/(d1[j]); 
		
		
		LAPf[cci] += (-f[l2]+16.0*f[l1]-30.0*f[i]+16.0*f[r1]-f[r2])/(d2[j]); 
	
	
	  }

		slip_rhs[ccif][0] =(f_s[0][cci]*f_s[1][cci]+f_s[1][cci]*f_s[2][cci]+f_s[2][cci]*f_s[0][cci])*(1.0+2.0*phi[cci])/(Mpl*Mpl);	
		slip_rhs[ccif][1] = 0.0;

		

		
	 }

   	}	
	
   }	
	
	//fprintf(fplin,"\n\n\n");
	
	fftw_execute(slip_plan_f);
	//printf("yha tk\n");
 	#pragma omp parallel for private(kfac)
	 for(ci[0]=2;ci[0]<(n_axis_loc[0]-2);++ci[0])
	  {
			
		for(ci[1]=2;ci[1]<(n_axis_loc[1]-2);++ci[1])
	  	{
		   for(ci[2]=2;ci[2]<(n_axis_loc[2]-2);++ci[2])
	  	  {
	  	   
			cci = ci[0]*(n_axis_loc[1]+4)*(n_axis_loc[2]+4) + ci[1]*(n_axis_loc[2]+4) + ci[2]+2;
			ccif = ci[0]*(n_axis_loc[1])*(n_axis_loc[2]) + ci[1]*(n_axis_loc[2]) + ci[2];

		kfac = tpie*tpie*(k_grid[cci][0]*k_grid[cci][1]+k_grid[cci][1]*k_grid[cci][2]+k_grid[cci][2]*k_grid[cci][0]);
	
		
		
		if(kfac>1e-14)
		{ slip_rhs_ft[ccif][0] = -slip_rhs_ft[ccif][0]/(kfac*n3sqrt); 
		  slip_rhs_ft[ccif][1] = -slip_rhs_ft[ccif][1]/(kfac*n3sqrt);

		}
		
		else
		{ 

		  slip_rhs_ft[ccif][0] = 0.0;
		  slip_rhs_ft[ccif][1] = 0.0;

		}

		
		

		
	 }
	}
   } 	


	fftw_execute(slip_plan_b);
	
	
	
	for(ci[0]=2;ci[0]<(n_axis_loc[0]-2);++ci[0])
	  {
			
		for(ci[1]=2;ci[1]<(n_axis_loc[1]-2);++ci[1])
	  	{
		   for(ci[2]=2;ci[2]<(n_axis_loc[2]-2);++ci[2])
	  	  {
	  	   
			cci = ci[0]*(n_axis_loc[1]+4)*(n_axis_loc[2]+4) + ci[1]*(n_axis_loc[2]+4) + ci[2];
			ccif = ci[0]*(n_axis_loc[1])*(n_axis_loc[2]) + ci[1]*(n_axis_loc[2]) + ci[2];
			
			tmpslip2[cci] = tmpslip1[cci];
		   tmpslip1[cci] = slip[cci]; 
		   slip[cci] = slip_rhs[ccif][0]/n3sqrt ; 


		slip_a[cci] = 0.5*(3.0*slip[cci]-4.0*tmpslip1[cci]+tmpslip2[cci])/da; 
	
	
		  }
		 }
	   }
	
	
	////////////////////////make data communication for slip///////////////////////
 MPI_Send((slip+strt),1,c_x_plain,up,01,cart_comm);
 MPI_Recv((slip+r_strt_l),1,c_x_plain,down,01,cart_comm,&stdn);
 
 MPI_Send((slip+strt_l),1,c_x_plain,down,00,cart_comm);
 MPI_Recv(slip+r_strt,1,c_x_plain,up,00,cart_comm,&stup);
	
	///////////////////////////////////////////////////////////////////////////////
	
	
	#pragma omp parallel for private(j,l1,l2,r1,r2,Vvl,V_fvl)
 for(ci[0]=2;ci[0]<(n_axis_loc[0]-2);++ci[0])
	  {
			
		for(ci[1]=2;ci[1]<(n_axis_loc[1]-2);++ci[1])
	  	{
		   for(ci[2]=2;ci[2]<(n_axis_loc[2]-2);++ci[2])
	  	  {
	  	   
			cci = ci[0]*(n_axis_loc[1]+4)*(n_axis_loc[2]+4) + ci[1]*(n_axis_loc[2]+4) + ci[2];
			
		

		
	
		


		  LAPslip[cci] = 0.0;
		 
	  	  
	    
	 	  wl = 1;
	   
	  for(j=2;j>-1;--j)
	  {	 
	
		l1 = cci - wl*(ci[j]) +  wl*(ci[j]-1) ;
		l2 = cci - wl*(ci[j]) +  wl*(ci[j]-2) ;
		r1 = cci - wl*(ci[j]) +  wl*(ci[j]+1) ;
		r2 = cci - wl*(ci[j]) +  wl*(ci[j]+2) ;


		wl*=(n_axis_loc[j]+4);


		LAPslip[cci] += (-slip[l2]+16.0*slip[l1]-30.0*slip[i]+16.0*slip[r1]-slip[r2])/(d2[j]); 
		
		
		tuldss[cci]+=  0.5*(1.0+2.0*phi[cci])*f_s[j][cci]*f_s[j][cci]/(ak*ak)  ;

		
	     }

//		fprintf(fplin,"%d\t%lf\n",i,LAPf[i]);
		Vvl = V(f[cci]);


		tuldss[cci]+=3.0*(Vvl - 0.5*tmpf_a[cci]*tmpf_a[cci]*a_t*a_t*(1.0-2.0*(phi[cci]-slip[cci])) - fbdss);

		
	}
	
   }	
  }

//	fprintf(fplin,"\n\n\n");
  


}





void cal_grd_tmunu()
{
	int cci,ccif,ci[3],l1,l2,r1,r2,j;
	double Vvl,V_fvl,fl;
	






	 #pragma omp parallel for private(j,l1,l2,r1,r2,Vvl,V_fvl,fl)
	 for(ci[0]=2;ci[0]<(n_axis_loc[0]-2);++ci[0])
	  {
			
		for(ci[1]=2;ci[1]<(n_axis_loc[1]-2);++ci[1])
	  	{
		   for(ci[2]=2;ci[2]<(n_axis_loc[2]-2);++ci[2])
	  	  {
	  	   
			cci = ci[0]*(n_axis_loc[1]+4)*(n_axis_loc[2]+4) + ci[1]*(n_axis_loc[2]+4) + ci[2]+2;
			ccif = ci[0]*(n_axis_loc[1])*(n_axis_loc[2]) + ci[1]*(n_axis_loc[2]) + ci[2];

	    
	     for(j=0;j<3;++j)
	     {	 
		

		
		tuldss[cci]+=  0.5*(1.0+2.0*phi[cci])*f_s[j][cci]*f_s[j][cci]/(a*a)  ;


	     }

			Vvl = V(f[cci]);
		V_fvl = V_f(f[cci]);
			
	
		


		fl = ( V_fvl/(a_t*a_t) + 3.0*f_a[cci]/a - 3.0*f_a[cci]*phi_a[cci] - 6.0*(phi[cci]-slip[cci])*f_a[cci]/a 
				- (phi_a[cci]-slip_a[cci])*f_a[cci]
				+(f_s[0][cci]*slip_s[0][cci]+f_s[1][cci]*slip_s[1][cci]+f_s[2][cci]*slip_s[2][cci])/(a*a*a_t*a_t) 
			)/(-1.0+2.0*(phi[cci]-slip[cci]))
			-a_tt*f_a[cci]/(a_t*a_t)  + 2.0*(LAPf[cci]/a)*(2.0*phi[cci]-slip[cci])/(a*a_t*a_t)  ; 

		scf_rhs[ccif][0] = f[cci] + da*f_a[cci] + 0.5*da*da*fl;	
		scf_rhs[ccif][1] = 0.0;
		



		tuldss[cci]+=3.0*(Vvl - 0.5*f_a[cci]*f_a[cci]*a_t*a_t*(1.0-2.0*(phi[cci]-slip[cci])) - fbdss);
		


		
  		


	  }
	 }
	}  
	

	fftw_execute(scf_plan_f);

}





int evolve(double aini, double astp)
{
    
	

   
    double ommi = omdmbini;
    double facb1,facb2,Vvl,V_fvl,fb_ak,fbk,omfb,Vvlb,V_fvlb,V_ffvlb,lin_delfac1,lin_delfac2,lin_phiac1,lin_phiac2,lin_delf_ak,lin_phi_ak;
    double w;

    int i,j,lcntr,cci,ci[3];

     ///Watch out for local vs global for parallelization
    double phiacc1[n*n*n],phiacc2[n*n*n],facc1[n*n*n],facc2[n*n*n],kfac2;
 


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

         
	  if((lcntr%jprint==0)&&(my_corank==0))
	  { 
		 

		fprintf(fpback,"%lf\t%.10lf\t%.10lf\n",a/ai,ommi*ai*ai*ai*Hi*Hi/(a*a_t*a_t),omfb*a*a/(a_t*a_t));
		fprintf(fplin,"%lf\t%.20lf\t%.20lf\n",a/ai,lin_growth,lin_phi);
		printf("a  %lf %.10lf  %.10lf\n",a,ommi*ai*ai*ai*Hi*Hi/(a*a_t*a_t),omfb);
		fflush(stdout);
	
		}

	
	if((lcntr%jprints==0))
	   { printf("printing..\n");

		// cal_dc();
      	//	 cal_spectrum(density_contrast,fppwspctrm_dc,0);
		// cal_spectrum(phi,fppwspctrm_phi,0);
		// write_fields();


	  }

		 
	#pragma omp parallel for private(i,Vvl,V_fvl,kfac2)
	 for(ci[0]=2;ci[0]<(n_axis_loc[0]-2);++ci[0])
	  {
			
		for(ci[1]=2;ci[1]<(n_axis_loc[1]-2);++ci[1])
	  	{
		   for(ci[2]=2;ci[2]<(n_axis_loc[2]-2);++ci[2])
	  	   {
/////////////////////phi acceleration calculation Step 1/////////////////////////////////////////////////////////////////////////////////

		//phiacc1 = (1.0/(a_t*a*a_t*a))*(- 2.0*a*phi[cci]*a_tt 
		//			      -a*a*tuldss[cci]/(6.0*Mpl*Mpl))  -phi[cci]/(a*a) 
		//				- 3.0*phi_a[cci]/a -phi_a[cci]/a - a_tt*phi_a[cci]/(a_t*a_t);

			cci = ci[0]*(n_axis_loc[1]+4)*(n_axis_loc[2]+4) + ci[1]*(n_axis_loc[2]+4) + ci[2]+2;
		
			phiacc1[cci] = (a_t*a_t/(a*a) + 2.0*a_tt/a )*(slip[cci]-phi[cci])/(a_t*a_t) + (slip_a[cci]-4.0*phi_a[cci])/a + (1.0/3.0)*LAPslip[cci]/(a*a*a_t*a_t)
					-(tuldss[cci])/(6.0*Mpl*Mpl*a_t*a_t)   - a_tt*phi_a[cci]/(a_t*a_t);


				V_fvl = V_f(f[cci]);
				Vvl = V(f[cci]);
	
			facc1[cci] = ( (V_fvl/(a_t*a_t) + 3.0*f_a[cci]/a - 3.0*f_a[cci]*phi_a[cci] - 6.0*(phi[cci]-slip[cci])*f_a[cci]/a 
				- (phi_a[cci]-slip_a[cci])*f_a[cci])/(-1.0+2.0*(phi[cci]-slip[cci]))
				+(f_s[0][cci]*slip_s[0][cci]+f_s[1][cci]*slip_s[1][cci]+f_s[2][cci]*slip_s[2][cci])/(a*a*a_t*a_t*(-1.0+2.0*(phi[cci]-slip[cci]))) 
					-(LAPf[cci]/a)*((1.0+2.0*phi[cci])/(-1.0+2.0*(phi[cci]-slip[cci])))/(a*a_t*a_t) )
			-a_tt*f_a[cci]/(a_t*a_t); 


		
			kfac2 = tpie*tpie*(k_grid[cci][0]*k_grid[cci][0]+k_grid[cci][1]*k_grid[cci][1]+k_grid[cci][2]*k_grid[cci][2]);
	    
	        scf_rhs_ft[cci][0] = (scf_rhs_ft[cci][0]/n3sqrt)/(1.0+0.5*kfac2*(da/(a_t*a))*(da/(a_t*a))); 
	        scf_rhs_ft[cci][1] = (scf_rhs_ft[cci][1]/n3sqrt)/(1.0+0.5*kfac2*(da/(a_t*a))*(da/(a_t*a)));

		//phiacc = (1.0/(2.0*a_t*a*a_t*a))*(-2.0*phi[cci]*a_t*a_t - 4.0*a*phi[cci]*a_tt 
			//		      -a*a*tuldss[cci]/(3.0*Mpl*Mpl)) - 3.0*phi_a[cci]/a -phi_a[cci]/a - a_tt*phi_a[cci]/(a_t*a_t);
		
		
			phi[cci]  = phi[cci]+da*phi_a[cci]+0.5*da*da*phiacc1[cci];
			tmpphi_a[cci] = phi_a[cci]+da*phiacc1[cci];

		//f[cci]  = f[cci]+da*f_a[cci]+0.5*da*da*facc1[cci];
			tmpf_a[cci] = f_a[cci]+da*facc1[cci];


			if(isnan(f[cci]+phi[cci]))
			{		fail=0;
				printf("field gone %lf %lf\n",f[cci],phi[cci]);
	
			}
			tul00[cci] = 0.0 ;
			tuldss[cci] = 0.0;
		
		
	  }

	 }
	}	
		
	 


		
	  ak = a + da;

	  Vvlb = V(fb);
          V_fvlb =  V_f(fb);
	  V_ffvlb =  V_ff(fb);

	  a_t = sqrt((ommi*Hi*Hi*ai*ai*ai/ak  + (1.0/(Mpl*Mpl))*ak*ak*Vvlb/(3.0*c)) / ( 1.0 - (1.0/(Mpl*Mpl))*ak*ak*fb_ak*fb_ak/(6.0*c*c*c))) ;
          a_tt = -0.5*ommi*Hi*Hi*ai*ai*ai/(ak*ak) - (1.0/(Mpl*Mpl*c))*ak*(fb_ak*fb_ak*a_t*a_t - Vvlb)/3.0;


	  fbdss =  (-0.5*fb_ak*fb_ak*a_t*a_t + Vvlb) ;
	  fb00 =  (0.5*fb_ak*fb_ak*a_t*a_t + Vvlb) ;
 
	   
/////////////////////Intermediate Tul calculations and psi construction//////////////////////////////////////////

	
	slip_fft_cal();
	
	



		
 
	




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
	{	fail=0;printf("%d Alert %lf\n",lcntr,facb2,fb_a);
		break;
		
	}



////////////////////////////Final Step//////////////////////////////////////////////////////////////////////////////////////
          
 	  
	  

	 
	#pragma omp parallel for private(i,Vvl,V_fvl)
	 for(ci[0]=2;ci[0]<(n_axis_loc[0]-2);++ci[0])
	  {
			
		for(ci[1]=2;ci[1]<(n_axis_loc[1]-2);++ci[1])
	  	{
		   for(ci[2]=2;ci[2]<(n_axis_loc[2]-2);++ci[2])
	  	  {
	  	   
			cci = ci[0]*(n_axis_loc[1]+4)*(n_axis_loc[2]+4) + ci[1]*(n_axis_loc[2]+4) + ci[2];	  	   
/////////////////////phi  acceleration calculation Final/////////////////////////////////////////////////////////////////////////////////
		
				phiacc2[cci] = (a_t*a_t/(ak*ak) + 2.0*a_tt/ak )*(slip[cci]-phi[cci])/(a_t*a_t) + (slip_a[cci]-4.0*tmpphi_a[cci])/ak 
				+ (1.0/3.0)*LAPslip[cci]/(ak*ak*a_t*a_t)
				-(tuldss[cci])/(6.0*Mpl*Mpl*a_t*a_t)   - a_tt*tmpphi_a[cci]/(a_t*a_t);


			V_fvl = V_f(f[cci]);
			Vvl = V(f[cci]);
	
			facc2[cci] = ( (V_fvl/(a_t*a_t) + 3.0*tmpf_a[cci]/ak - 3.0*tmpf_a[cci]*tmpphi_a[cci] - 6.0*(phi[cci]-slip[cci])*tmpf_a[cci]/ak 
				- (tmpphi_a[cci]-slip_a[cci])*tmpf_a[cci])/(-1.0+2.0*(phi[cci]-slip[cci])) 
				+(f_s[0][cci]*slip_s[0][cci]+f_s[1][cci]*slip_s[1][cci]+f_s[2][cci]*slip_s[2][cci])/(ak*ak*a_t*a_t*(-1.0+2.0*(phi[cci]-slip[cci]))) 
					-(LAPf[cci]/ak)*((1.0+2.0*phi[cci])/(-1.0+2.0*(phi[cci]-slip[cci])))/(ak*a_t*a_t) )
					-a_tt*tmpf_a[cci]/(a_t*a_t); 
	



		
		
			phi_a[cci] = phi_a[cci]+0.5*da*(phiacc1[cci]+phiacc2[cci]);
			f_a[cci] = f_a[cci]+da*(facc2[cci]);
		//f_a[cci] = (f_a[cci]+da*(facc2[cci]))/(1.0 - da*( (3.0/ak - 3.0*phi_a[cci] - 6.0*(phi[cci]-slip[cci])/ak 
		//		- (phi_a[cci]-slip_a[cci]))/(-1.0+2.0*(phi[cci]-slip[cci])) -a_tt/(a_t*a_t)) );

 			if(isnan(phi[cci]+phi_a[cci]))
			{fail=0; printf("phi_gone  %d  phiacc1  %lf  phiacc2  %lf\n",ci,phiacc1[cci],phiacc2[cci]);
			}
			tul00[cci] = 0.0 ;
			tuldss[cci] = 0.0;

	  }
	}
   }
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////Final Tul and phi recosntruction/////////////////////////////////////////////////////////////
	
	 a = a+da;
	
      a_t = sqrt((Hi*Hi*ommi*ai*ai*ai/a  + (1.0/(Mpl*Mpl))*a*a*Vvlb/(3.0*c)) / ( 1.0 - (1.0/(Mpl*Mpl))*a*a*fb_a*fb_a/(6.0*c*c*c))) ;
      a_tt = -0.5*ommi*Hi*Hi*ai*ai*ai/(a*a) - (1.0/(Mpl*Mpl*c))*a*(fb_a*fb_a*a_t*a_t - Vvlb)/3.0;

	fbdss =  (-0.5*fb_a*fb_a*a_t*a_t + Vvlb) ;
	fb00 =  (0.5*fb_a*fb_a*a_t*a_t + Vvlb) ;

	


	

	cal_grd_tmunu();
	
		
	


	
	
	

     

 //   printf("evolve w  %.10lf  Hi %.10lf  %.10lf  %.10lf\n",a_t,a,a0);

    if(fail!=1)
    {printf("fail  %d %d  %lf\n",fail,lcntr,a); 
	return(fail);
    }    
	
  }
 return(fail);
}






