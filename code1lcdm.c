#include <fftw3.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h> 
#include <omp.h>

#define  n 16

double G   = 1.0;
double c   = 1.0;
double Mpl ;
double H0  = 22.04*(1e-5);
double L[3];
int N;


double *psi, *phi, *f,*psi_a, *phi_a, *f_a,*tul00,*tuldss;
double psi_s[3][n*n*n],phi_s[3][n*n*n],f_s[3][n*n*n],LAPphi[n*n*n],LAPpsi[n*n*n],LAPf[n*n*n],usty[n*n*n],psty[n*n*n];
double *tmppsi, *tmpphi, *tmpf,*tmppsi_a, *tmpphi_a, *tmpf_a,m;
double dx[3];
struct particle
	{	
		double x[3];
		double v[3];
		
		
		int 	cubeind[8];	

	};



struct particle p[n*n*n],tmpp[n*n*n];
double grid[n*n*n][3];
int gridind[n*n*n][3];

int nic[n*n*n][16];

double fb, fb_a, omdmb, omdeb, a, at, a_t, a_tt, Vamp, ai, a0, da, ak, fbt, fb_at;
int jprint;
double H0, Hi;

FILE *fpback;


void background();

double V(double);
double V_f(double);


void initialise();

double mesh2particle(struct particle *,int,double *);
void particle2mesh(struct particle * ,int ,double *,double *,double );
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
	//f = (double *) malloc(n*n*n*sizeof(double)); 
        //f_a = (double *) malloc(n*n*n*sizeof(double)); 
	tul00 = (double *) malloc(n*n*n*sizeof(double)); 
        tuldss = (double *) malloc(n*n*n*sizeof(double));

	tmppsi = (double *) malloc(n*n*n*sizeof(double)); 
        tmppsi_a = (double *) malloc(n*n*n*sizeof(double)); 
	tmpphi = (double *) malloc(n*n*n*sizeof(double)); 
        tmpphi_a = (double *) malloc(n*n*n*sizeof(double)); 
	//tmpf = (double *) malloc(n*n*n*sizeof(double)); 
        //tmpf_a = (double *) malloc(n*n*n*sizeof(double)); 
 

	//m = (double *) malloc(n*n*n*sizeof(double)); 
	

      

	background();
	initialise();

       i = evolve(ai,a0);
      




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
			
		
		rv+= del*meshf[k];



	}
	

	return(rv);

}


void particle2mesh(struct particle * pp,int p_id,double *meshpsi,double *meshphi,double ap)
{

	int i,j,k;
	int anchor[3];
	double rvphi=0.0,rvpsi=0.0,del[8];
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
		rvpsi+= del[i]*meshpsi[k];
	}	
	for(i=0;i<8;++i)
	{
		k = pp[p_id].cubeind[i];
		tul00[k]+= m*del[i]*(1.0+3.0*rvphi-rvpsi-gamma*gamma*(vmgsqr*a*a*rvphi+rvpsi))/(ap*ap*ap);
		tuldss[k]+= (vmgsqr/3.0)*del[i]*(1.0+3.0*rvphi-rvpsi-gamma*gamma*(vmgsqr*a*a*rvphi+rvpsi))/(ap*ap*ap);
		psty[k]+= sqrt(vmgsqr)*m*del[i]*(1.0+3.0*rvphi-gamma*gamma*vmgsqr*a*a*rvphi)/(ap*ap*ap);
		usty[k]+= m*del[i]*gamma*(-1.0-gamma*gamma)/(6.0*a_t*a_t*Mpl*Mpl*ap);

		 

	}

	
	

	

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
      int l1,l2,r1,r2,Vvl,V_fvl;

      double cpmc = (0.14/(0.68*0.68));
      int px,py,pz,ci,pgi,j;
      double gamma, v, gradmagf;
      a0 = 1000.0;
      ai = 1.0;
      fb = 1.0;
      fb_a = 0.0;
      omdmb= (cpmc)*pow((a0/ai),3.0)/(cpmc*a0*a0*a0/(ai*ai*ai) + (1.0-cpmc));

	dx[0] = 0.001; dx[1] =0.001; dx[2] = 0.001;
        L[0] = dx[0]*((double) n);  L[1] = dx[1]*((double) n);  L[2] = dx[2]*((double) n);
      
	for(ci = 0;ci <N; ++ci)
	{
		
		phi[ci] = 0.01*((double) rand()/((double) RAND_MAX  ));
		psi[ci] = phi[ci];
		f[ci]  =  fb;
		f_a[ci]  =  fb_a;
      		phi_a[ci] = 0.0;
		psi_a[ci] = 0.0;
		for(j=0;j<3;++j)
		{	p[ci].x[j] =  ((double) rand()/((double) RAND_MAX  ))*L[j];
			p[ci].v[j] = 0.0;
		}
		
				
				
      	}


	#pragma omp parallel for
	  for(ci=0;ci<N;++ci)
	  {
	    particle2mesh(p,ci,psi,phi,a);

	    usty[ci]+= 1.0-(at*at/(Mpl*Mpl*6.0))*f_a[ci]*f_a[ci];

	    Vvl = V(f[ci]);

	    LAPf[ci] = 0.0;
	    LAPpsi[ci] = 0.0;
	    LAPphi[ci] = 0.0;	
	    for(j=0;j<3;++j)
	     {	 
		l1 = (N+ci-((int)(pow(n,2-j))))%N;
		l2 = (N+ci-2*((int)(pow(n,2-j))))%N;
		r1 = (N+ci+((int)(pow(n,2-j))))%N;
		r2 = (N+ci+2*((int)(pow(n,2-j))))%N;

		f_s[j][ci] = (f[l2]-8.0*f[l1]+8.0*f[r1]-f[r2])/(12.0*dx[j]);
		
		phi_s[j][ci] = (phi[l2]-8.0*phi[l1]+8.0*phi[r1]-phi[r2])/(12.0*dx[j]); 
		psi_s[j][ci] = (psi[l2]-8.0*psi[l1]+8.0*psi[r1]-phi[r2])/(12.0*dx[j]); 
		
		LAPphi[ci] += (-phi[l2]+16.0*phi[l1]-30.0*phi[ci]+16.0*phi[r1]-phi[r2])/(12.0*dx[j]*dx[j]); 
		LAPpsi[ci] += (-psi[l2]+16.0*psi[l1]-30.0*psi[ci]+16.0*psi[r1]-psi[r2])/(12.0*dx[j]*dx[j]); 

		LAPf[ci] += (-f[l2]+16.0*f[l1]-30.0*f[ci]+16.0*f[r1]-f[r2])/(12.0*dx[j]*dx[j]); 
		 psty[ci]+=0.5*f_s[j][ci]*f_s[j][ci]*(1.0+2.0*phi[ci])/(a*a);
	     }
  	    psty[ci]+=Vvl + 0.5*f_a[ci]*a_t*f_a[ci]*a_t; 	


	  }
	           
          
	



	  

}


int evolve(double aini, double astp)
{
    


    double ommi = omdmb;
    double facb;
    double w;

    int fail = 1,i,j,ci;

    double nd = (double) n, jd;  ///Watch out for local vs global for parallelization
    double phiacc,facc,pacc[3],Vvl,V_fvl,v,gamma,phiavg,psiavg,phi_aavg,psi_aavg,psi_savg[3],phi_savg[3],fsg,psiold;
    double vmagsqr;
    int anchor[3];
    double lplphi,lplpsi,lplf;
    int l2,l1,r1,r2;
    

    for(a=aini,i=0;(a<=astp)&&(fail==1);++i)
	{ if(i%jprint==0)
	   printf("a  %lf\n",a);
          
          Vvl = V(fb);
  	  V_fvl = V_f(fb);
	  a_t = sqrt((ommi*ai*ai*ai/a  + (1.0/(Mpl*Mpl))*a*a*Vvl/(3.0*c)) / ( 1.0 - (1.0/(Mpl*Mpl))*a*a*fb_a*fb_a/(6.0*c*c*c))) ;
          a_tt = -0.5*ommi*ai*ai*ai/(a*a) - (1.0/(Mpl*Mpl*c))*a*(fb_a*fb_a*a_t*a_t - Vvl)/3.0;
          facb = -V_fvl*c*c/(a_t*a_t) - 3.0*fb_a/a - a_tt*fb_a/(a_t*a_t);
	  
	  fbt = fb + 0.5*da*fb_a;
	  fb_at = fb_a + 0.5*da*facb;
	  at = a + 0.5*da;

/////////////////////////////////particle force calculation*****Step 1////////////////////////////////////////////////		 

	 for(ci=0;ci<N;++ci)
	  {
			vmagsqr = 0.0;	

	
		phiavg = mesh2particle(p,ci,phi);
		psiavg = mesh2particle(p,ci,psi);
		phi_aavg = mesh2particle(p,ci,phi_a);
		psi_aavg = mesh2particle(p,ci,psi_a);
		fsg = 0.0;
		for(i=0;i<3;++i)
		{	psi_savg[i] = mesh2particle(p,ci,&psi_s[i][0]);
			phi_savg[i] = mesh2particle(p,ci,&phi_s[i][0]);
			fsg+= 2.0*p[ci].v[i]*a_t*( phi_savg[i] + psi_savg[i] );
			vmagsqr+=p[ci].v[i]*p[ci].v[i];

		}
			gamma = 1.0/sqrt(1.0-a*a*a_t*a_t*vmagsqr);
		for(i=0;i<3;++i)
		{
			
			pacc[i] = (p[ci].v[i]*a_t*a_t*a_t*vmagsqr*(-2.0*a*a_t*(phiavg+psiavg)-a*a*phi_aavg*a_t+a*a_t)
				 +p[ci].v[i]*a_t*(fsg + psi_aavg*a_t -2.0*a_t/a + 2.0*psi_aavg*a_t -phi_savg[i])
				 -psi_savg[i]/(a))/(a*a) - a_tt*p[ci].v[i]/(a_t*a_t);
			tmpp[ci].x[i] = p[ci].x[i] + 0.5*da*p[ci].v[i];
			tmpp[ci].v[i] = p[ci].v[i] + 0.5*da*pacc[i]; 
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

		phiacc = (1.0/(a_t*a*a_t*a))*(-2.0*psi[ci]*a_t*a_t - 4.0*a*psi[ci]*a_tt + (2.0/3.0)*(LAPphi[ci]-LAPpsi[ci])
					      -a*a*tuldss[ci]/(3.0*Mpl*Mpl))/(a_t*a_t) - 3.0*phi_a[ci]/a -psi_a[ci]/a - a_tt*phi_a[ci]/(a_t*a_t);

		Vvl = V(f[ci]);
  	  	V_fvl = V_f(f[ci]);
		facc   = (1.0/(a*a*(-1.0+2.0*psi[ci])))*( (a*a*V_fvl - f_s[ci][0]*(phi_s[ci][0]-psi_s[ci][0])
								- f_s[ci][2]*(phi_s[ci][2]-psi_s[ci][2])- f_s[ci][2]*(phi_s[ci][2]-psi_s[ci][2])
							  -LAPf[ci]*(1.0+2.0*phi[ci]))/(a_t*a_t) -3.0*a*f_a[ci] -6.0*a*psi[ci]*f_a[ci]
							   -a*a*f_a[ci]*psi[ci] ) - a_tt*f_a[ci]/(a_t*a_t);
		tmpphi[ci]  = phi[ci]+0.5*da*phi_a[ci];
		tmpphi_a[ci] = phi_a[ci]+0.5*da*phiacc;
		tmpf[ci]  = f[ci]+0.5*da*f_a[ci];
		tmpf_a[ci] = f_a[ci]+0.5*da*facc;
	  }
		
	

	   Vvl = V(fbt);
  	  V_fvl = V_f(fbt);
	  a_t = sqrt((ommi*ai*ai*ai/at  + (1.0/(Mpl*Mpl))*at*at*Vvl/(3.0*c)) / ( 1.0 - (1.0/(Mpl*Mpl))*at*at*fb_at*fb_at/(6.0*c*c*c))) ;
          a_tt = -0.5*ommi*ai*ai*ai/(at*at) - (1.0/(Mpl*Mpl*c))*at*(fb_at*fb_at*a_t*a_t - Vvl)/3.0;
          facb = -V_fvl*c*c/(a_t*a_t) - 3.0*fb_at/a - a_tt*fb_at/(a_t*a_t);
	  
	  fb = fb + da*fb_a;
	  fb_a = fb_a + da*facb;
/////////////////////Intermediate Tul calculations and Psi construction//////////////////////////////////////////
	  #pragma omp parallel for
	  for(ci=0;ci<N;++ci)
	  {
	    particle2mesh(tmpp,ci,tmppsi,tmpphi,at);

	    usty[ci]+= 1.0-(at*at/(Mpl*Mpl*6.0))*f_a[ci]*f_a[ci];

	    Vvl = V(f[ci]);

	    LAPf[ci] = 0.0;
	    LAPpsi[ci] = 0.0;
	    LAPphi[ci] = 0.0;	
	    for(j=0;j<3;++j)
	     {	 
		l1 = (N+ci-((int)(pow(n,2-j))))%N;
		l2 = (N+ci-2*((int)(pow(n,2-j))))%N;
		r1 = (N+ci+((int)(pow(n,2-j))))%N;
		r2 = (N+ci+2*((int)(pow(n,2-j))))%N;

		f_s[j][ci] = (tmpf[l2]-8.0*tmpf[l1]+8.0*tmpf[r1]-tmpf[r2])/(12.0*dx[j]);
		
		phi_s[j][ci] = (tmpphi[l2]-8.0*tmpphi[l1]+8.0*tmpphi[r1]-tmpphi[r2])/(12.0*dx[j]); 
		
		LAPphi[ci] += (-tmpphi[l2]+16.0*tmpphi[l1]-30.0*tmpphi[ci]+16.0*tmpphi[r1]-tmpphi[r2])/(12.0*dx[j]*dx[j]); 

		LAPf[ci] += (-tmpf[l2]+16.0*tmpf[l1]-30.0*tmpf[ci]+16.0*tmpf[r1]-tmpf[r2])/(12.0*dx[j]*dx[j]); 
		 psty[ci]+=0.5*f_s[j][ci]*f_s[j][ci]*(1.0+2.0*tmpphi[ci])/(at*at);
	     }
  	    psty[ci]+=Vvl + 0.5*f_a[ci]*a_t*f_a[ci]*a_t; 	


	  }
	  #pragma omp parallel for
	  for(ci=0;ci<N;++ci)
	  {
		tmppsi[ci] = (1.0/usty[ci])*( LAPphi[ci]/(3.0*a_t*a_t) - (at*at)*psty[ci]/(Mpl*Mpl*6.0*a_t*a_t) +0.5);
		tmppsi_a[ci] = 2.0*(tmppsi[ci] - psi[ci])/da;
		
		
                                      
          }
	#pragma omp parallel for
	  for(ci=0;ci<N;++ci)
	  {
 		LAPphi[ci] = 0.0;
		for(j=0;j<3;++j)
	     {	 
		l1 = (N+ci-((int)(pow(n,2-j))))%N;
		l2 = (N+ci-2*((int)(pow(n,2-j))))%N;
		r1 = (N+ci+((int)(pow(n,2-j))))%N;
		r2 = (N+ci+2*((int)(pow(n,2-j))))%N;

		
		
		psi_s[j][ci] = (tmppsi[l2]-8.0*tmpphi[l1]+8.0*tmpphi[r1]-tmpphi[r2])/(12.0*dx[j]); 
		
		LAPphi[ci] += (-tmpphi[l2]+16.0*tmpphi[l1]-30.0*tmpphi[ci]+16.0*tmpphi[r1]-tmpphi[r2])/(12.0*dx[j]*dx[j]); 

		
	     }



	  }





/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// 


////////////////////////////Final Step//////////////////////////////////////////////////////////////////////////////////////
          
 	  
	  

	 

	 for(ci=0;ci<N;++ci)
	  {
			vmagsqr = 0.0;	

		
		phiavg = mesh2particle(tmpp,ci,tmpphi);
		psiavg = mesh2particle(tmpp,ci,tmppsi);
		phi_aavg = mesh2particle(tmpp,ci,tmpphi_a);
		psi_aavg = mesh2particle(tmpp,ci,tmppsi_a);
		fsg = 0.0;
		for(i=0;i<3;++i)
		{	psi_savg[i] = mesh2particle(tmpp,ci,&psi_s[i][0]);
			phi_savg[i] = mesh2particle(tmpp,ci,&phi_s[i][0]);
			fsg+= 2.0*tmpp[ci].v[i]*a_t*( phi_savg[i] + psi_savg[i] );
			vmagsqr+=tmpp[ci].v[i]*tmpp[ci].v[i];

		}
			gamma = 1.0/sqrt(1.0-at*at*a_t*a_t*vmagsqr);
		for(i=0;i<3;++i)
		{
			
			pacc[i] = (tmpp[ci].v[i]*a_t*a_t*a_t*vmagsqr*(-2.0*at*a_t*(phiavg+psiavg)-at*at*phi_aavg*a_t+a*a_t)
				 +tmpp[ci].v[i]*a_t*(fsg + psi_aavg*a_t -2.0*a_t/at + 2.0*psi_aavg*a_t -phi_savg[i])
				 -psi_savg[i]/(at))/(at*at) - a_tt*tmpp[ci].v[i]/(a_t*a_t);
			p[ci].x[i] = p[ci].x[i] + 0.5*da*p[ci].v[i];
			p[ci].v[i] = p[ci].v[i] + 0.5*da*pacc[i]; 
			anchor[i] =  ( n + (int) (tmpp[ci].x[i]/dx[i]))%n;
	

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

		phiacc = (1.0/(a_t*at*a_t*at))*
				(-2.0*tmppsi[ci]*a_t*a_t - 4.0*at*tmppsi[ci]*a_tt + (2.0/3.0)*(LAPphi[ci]-LAPpsi[ci])
				-at*at*tuldss[ci]/(3.0*Mpl*Mpl))/(a_t*a_t) - 3.0*tmpphi_a[ci]/at -tmppsi_a[ci]/a - a_tt*tmpphi_a[ci]/(a_t*a_t);

		Vvl = V(tmpf[ci]);
  	  	V_fvl = V_f(tmpf[ci]);
		facc   = (1.0/(at*at*(-1.0+2.0*tmppsi[ci])))*( (at*at*V_fvl - f_s[ci][0]*(phi_s[ci][0]-psi_s[ci][0])
					- f_s[ci][2]*(phi_s[ci][2]-psi_s[ci][2])- f_s[ci][2]*(phi_s[ci][2]-psi_s[ci][2])
					-LAPf[ci]*(1.0+2.0*tmpphi[ci]))/(a_t*a_t) -3.0*at*tmpf_a[ci] -6.0*at*tmppsi[ci]*tmpf_a[ci]
							   -at*at*tmpf_a[ci]*tmppsi[ci] ) - a_tt*f_a[ci]/(a_t*a_t);
		phi[ci]  = phi[ci]+0.5*da*phi_a[ci];
		phi_a[ci] = phi_a[ci]+0.5*da*phiacc;
		f[ci]  = f[ci]+0.5*da*f_a[ci];
		f_a[ci] = f_a[ci]+0.5*da*facc;




	}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////Final Tul and Psi recosntruction/////////////////////////////////////////////////////////////
	a = a+ da;	


	  #pragma omp parallel for
	  for(ci=0;ci<N;++ci)
	  {
	    particle2mesh(p,ci,psi,phi,a);

	    usty[ci]+= 1.0-(at*at/(Mpl*Mpl*6.0))*f_a[ci]*f_a[ci];

	    Vvl = V(f[ci]);

	    LAPf[ci] = 0.0;
	    LAPpsi[ci] = 0.0;
	    LAPphi[ci] = 0.0;	
	    for(j=0;j<3;++j)
	     {	 
		l1 = (N+ci-((int)(pow(n,2-j))))%N;
		l2 = (N+ci-2*((int)(pow(n,2-j))))%N;
		r1 = (N+ci+((int)(pow(n,2-j))))%N;
		r2 = (N+ci+2*((int)(pow(n,2-j))))%N;

		f_s[j][ci] = (f[l2]-8.0*f[l1]+8.0*f[r1]-f[r2])/(12.0*dx[j]);
		
		phi_s[j][ci] = (phi[l2]-8.0*phi[l1]+8.0*phi[r1]-phi[r2])/(12.0*dx[j]); 
		
		LAPphi[ci] += (-phi[l2]+16.0*phi[l1]-30.0*phi[ci]+16.0*phi[r1]-phi[r2])/(12.0*dx[j]*dx[j]); 

		LAPf[ci] += (-f[l2]+16.0*f[l1]-30.0*f[ci]+16.0*f[r1]-f[r2])/(12.0*dx[j]*dx[j]); 
		 psty[ci]+=0.5*f_s[j][ci]*f_s[j][ci]*(1.0+2.0*phi[ci])/(a*a);
	     }
  	    psty[ci]+=Vvl + 0.5*f_a[ci]*a_t*f_a[ci]*a_t; 	


	  }
	  #pragma omp parallel for
	  for(ci=0;ci<N;++ci)
	  {	psiold = psi[ci];
		psi[ci] = (1.0/usty[ci])*( LAPphi[ci]/(3.0*a_t*a_t) - (a*a)*psty[ci]/(Mpl*Mpl*6.0*a_t*a_t) +0.5);
		psi_a[ci] = 2.0*(psi[ci]-tmppsi[ci])/da;
		
		
                                      
          }
	#pragma omp parallel for
	  for(ci=0;ci<N;++ci)
	  {
 		LAPpsi[ci] = 0.0;
		for(j=0;j<3;++j)
	     {	 
		l1 = (N+ci-((int)(pow(n,2-j))))%N;
		l2 = (N+ci-2*((int)(pow(n,2-j))))%N;
		r1 = (N+ci+((int)(pow(n,2-j))))%N;
		r2 = (N+ci+2*((int)(pow(n,2-j))))%N;

		
		
		psi_s[j][ci] = (psi[l2]-8.0*phi[l1]+8.0*phi[r1]-phi[r2])/(12.0*dx[j]); 
		
		LAPpsi[ci] += (-psi[l2]+16.0*psi[l1]-30.0*psi[ci]+16.0*psi[r1]-psi[r2])/(12.0*dx[j]*dx[j]); 

		
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

}






