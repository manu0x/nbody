#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h> 
#include <omp.h>
#include <mpi.h>
#include <fenv.h>
#include <time.h>


#define  n 10

#define tpie  2.0*M_PI

FILE *fppwspctrm;

/////////// MPI related ////////////////////


int mpicheck = 0;
int num_p = 0;
int rank;


int nd_cart;
int *my_coords;
int my_corank;
int n_axis_loc[2];

MPI_Datatype c_x_plain,c_y_plain;

MPI_Comm cart_comm;


//////////////////////////////////////////////


double G   = 1.0;
double c   = 1.0;
double Mpl ;
double lenfac = 1.0;
double Hb0  ;
double L[2];
int tN;
int n_axis[2];
int fail =1;

clock_t t_start,t_end;

double n3sqrt;
double *phi, *phi_a,  *f,*f_a,*slip,*slip_a,*tul00,*tuldss,fbdss,fb00;
double *phi_s[2],*f_s[2],*slip_s[2],*LAPslip,*LAPf,*tmpslip2,*tmpslip1;
double *tmpphi,  *tmpf,*tmpphi_a, *tmpf_a, *ini_vel0,*ini_vel1,*ini_vel2,m=1.0;
double *dx,*d1,*d2;
double *density_contrast,*ini_density_contrast,*ini_phi_potn;


void allocate_fields(int nax[2])
{

	 	
	
	int l = (nax[0]+2)*(nax[1]+2);
	
	phi  = calloc(l,sizeof(double));
	


	

	LAPslip  = calloc(l,sizeof(double));
	LAPf = calloc(l,sizeof(double));
	tmpslip1  = calloc(l,sizeof(double));
	tmpslip2  = calloc(l,sizeof(double));

	density_contrast = calloc(l,sizeof(double));


	printf("Allocated...%d\n",l);


}
/*struct particle
	{	
		double x[2];
		double v[2];
		
		
		int 	cubeind[8];	

	};



struct particle p[n*n*n],tmpp[n*n*n];
*/
double *grid[2];
int *ind_grid[2];
double *k_grid[2];
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






void main(int argc, char **argv)
{   t_start = clock();


	int i;
      
	n_axis[0]=n;
	n_axis[1]=n;
	


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

	for(i=0;i<2;++i)
	{	
		if(i<nd_cart)
		{
		   double tmp_naxis = (((double) n_axis[i])/ ((double) dims[i]));
		   n_axis_loc[i] = (int) tmp_naxis;


		    if((n_axis[i]%dims[i]) != 0)
		     {
		     	n_axis_loc[i] +=1;
			 if(my_coords[i]==(dims[i]-1))
		      		n_axis_loc[i]+=(n_axis[i]-dims[i]*n_axis_loc[i]);

		      }

		}

		else
			n_axis_loc[i] = n_axis[i]; 


	}
	
	
  MPI_Type_vector(n_axis_loc[1],1,n_axis_loc[0]+2,MPI_DOUBLE,&c_x_plain);
  MPI_Type_commit(&c_x_plain);
  MPI_Type_vector(n_axis_loc[0],1,1,MPI_DOUBLE,&c_y_plain);
  MPI_Type_commit(&c_y_plain);

	

	


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
	
	printf("\tco \t%d\n",my_corank);
	//printf("nx  %d ny  %d nz  %d\n",n_axis_loc[0],n_axis_loc[1],n_axis_loc[2]);
	//printf("Local\n");
	//printf("nx  %d ny  %d nz  %d\n\n\n",n_axis_loc[0],n_axis_loc[1],n_axis_loc[2]);

	

}


void printing2()
{	
	int i;
	//printf("Pocess  %d\n coords \n",rank);




	for(i=0;i<2;++i)
	{
		printf("%d\t",n_axis_loc[i]);


	}
	
	printf("\tco2 \t%d\n",my_corank);
	//printf("nx  %d ny  %d nz  %d\n",n_axis_loc[0],n_axis_loc[1],n_axis_loc[2]);
	//printf("Local\n");
	//printf("nx  %d ny  %d nz  %d\n\n\n",n_axis_loc[0],n_axis_loc[1],n_axis_loc[2]);

	

}



void write_data()
{
	int i,j,ci,pr,up,down,left,right;
	FILE *fp;
	MPI_Status stup,stdn,stlt,strt;
	for(i=1;i<n_axis_loc[0]+1;++i)
	{
			for(j=1;j<n_axis_loc[1]+1;++j)
			{
				ci = (n_axis_loc[1]+2)*i + j;
				if(ci>1155)
				printf("ALLLETS %d %d %d\n",i,j,ci);
			
				phi[ci] = (double) ((i-1)*n_axis_loc[1]+(j-1));
	
			}	
	}




 for(pr = 0;pr<num_p;++pr)
 {
 
 	MPI_Barrier(cart_comm);
 	if(pr==my_corank)
	{ if(my_corank==0)
	  fp = fopen("tmp.txt","w");
	  else
	  fp = fopen("tmp.txt","a");
	
	  printf("Writing process %d\n",my_corank);
	
	  for(i=0;i<n_axis_loc[0]+2;++i)
	  {
			for(j=0;j<n_axis_loc[1]+2;++j)
			{	ci = (n_axis_loc[1]+2)*i + j;
				fprintf(fp,"%lf\t",phi[ci]);
	
			}
			
			fprintf(fp,"\n");	
	 }
	 fprintf(fp,"\n\n\n");
	 fclose(fp);
  }
 }
 
 MPI_Cart_shift(cart_comm,0,1,&left,&right)
 MPI_Cart_shift(cart_comm,1,1,&down,&up)
 
 MPI_Send(&phi[1],1,c_y_plain,up,10,);
 MPI_Recv(&phi[0],1,c_y_plain,up,11,&stup);
 
 MPI_Send(&(phi[1]+(n_axis_loc[0]+2)*(n_axis_loc[1])),1,c_y_plain,down,11);
 MPI_Recv(&(phi[0]+(n_axis_loc[0]+2)*(n_axis_loc[1])),1,c_y_plain,down,10,&stdn);
 
 MPI_Send(&(phi[0]+n_axis_loc[0]),1,c_x_plain,right,00);
 MPI_Recv(&phi[0],1,c_x_plain,right,00);
 
 MPI_Send(&(phi[1]),1,c_x_plain,left,01);
 
 

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


allocate_fields(n_axis_loc);
write_data();


MPI_Finalize();


	
	
	
	
	


}








