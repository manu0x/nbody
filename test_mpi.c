#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h> 
#include <omp.h>
#include <mpi.h>
#include <fenv.h>
#include <time.h>


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
double *dx,*d1,*d2;
double *density_contrast,*ini_density_contrast,*ini_phi_potn;


void allocate_fields(int nax[3],int ndc)
{

	 int nax_l[3];
	 int il;	
	
	int l = (nax[10+2)*(nax[1]+2)*(nax[2]+2);
	
	phi  = calloc(l,sizeof(double));
	phi_a  = calloc(l,sizeof(double));
	f  = calloc(l,sizeof(double));





}
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




void main(int argc, char **argv)
{   t_start = clock();


	int i;
      
	n_axis[0]=n;
	n_axis[1]=n;
	n_axis[2]=n;


	mpicheck = MPI_Init(&argc,&argv);
	mpicheck = MPI_Comm_size(MPI_COMM_WORLD,&num_p);
	mpicheck = MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	if(num_p<4)
		nd_cart = 1;
	else
	{	if(num_p<20)
			nd_cart = 2;
		else
			nd_cart = 3;
	}

	int * dims = calloc(nd_cart,sizeof(int));
	int * periods = calloc(nd_cart,sizeof(int));
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
		   double tmp_naxis = (((double) n_axis[i])/ ((double) dims[i]));
		   n_axis_loc[i] = (int) tmp_naxis;


		    if((n_axis[i]%dims[i]) != 0)
		     {
		     
			 if(my_coords[i]==(dims[i]-1))
		      		n_axis_loc[i]+=(n_axis[i]-dims[i]*n_axis_loc[i]);

		      }

		}

		else
			n_axis_loc[i] = n_axis[i]; 


	}

	

	


void printing()
{	
	int i;
	//printf("Pocess  %d\n coords \n",rank);

	if(rank==0)
	{ 
		printf("dims are \n");
	for(i=0;i<nd_cart;++i)
	 {
		printf("%d\t",dims[i]);


	 }

		printf("\n\n");
	}



	for(i=0;i<nd_cart;++i)
	{
		printf("%d\t",my_coords[i]);


	}
	
	printf("\t \t%d\n",my_corank);
	//printf("nx  %d ny  %d nz  %d\n",n_axis_loc[0],n_axis_loc[1],n_axis_loc[2]);
	//printf("Local\n");
	//printf("nx  %d ny  %d nz  %d\n\n\n",n_axis_loc[0],n_axis_loc[1],n_axis_loc[2]);

	

}


void printing2()
{	
	int i;
	//printf("Pocess  %d\n coords \n",rank);




	for(i=0;i<3;++i)
	{
		printf("%d\t",n_axis_loc[i]);


	}
	
	printf("\t \t%d\n",my_corank);
	//printf("nx  %d ny  %d nz  %d\n",n_axis_loc[0],n_axis_loc[1],n_axis_loc[2]);
	//printf("Local\n");
	//printf("nx  %d ny  %d nz  %d\n\n\n",n_axis_loc[0],n_axis_loc[1],n_axis_loc[2]);

	

}





for(i=0;i<num_p;++i)
{
	MPI_Barrier(cart_comm);
	if(my_corank==i)
  	{printing();
		 }

}
	


MPI_Barrier(cart_comm);



for(i=0;i<num_p;++i)
{
	MPI_Barrier(cart_comm);
	if(my_corank==0&&(i==0))
	printf("\n");
	
	if(my_corank==i)
  	{printing2();
		 }

}


MPI_Finalize();


	
	
	
	
	


}








