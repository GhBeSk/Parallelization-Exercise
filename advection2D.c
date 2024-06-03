/*******************************************************************************
2D advection example program which advects a Gaussian u(x,y) at a fixed velocity



Outputs: initial.dat - inital values of u(x,y) 
         final.dat   - final values of u(x,y)

         The output files have three columns: x, y, u

         Compile with: gcc -o advection2D -std=c99 advection2D.c -lm

Notes: The time step is calculated using the CFL condition

********************************************************************************/

/*********************************************************************
                     Include header files 
**********************************************************************/

#include <stdio.h>
#include <math.h>

/*********************************************************************
                      Main function
**********************************************************************/

int main(){

  /* Grid properties */
  const int NX=1000;    // Number of x points
  const int NY=1000;    // Number of y points
  const float xmin=0.0; // Minimum x value
  const float xmax=30.0; // Maximum x value
  const float ymin=0.0; // Minimum y value
  const float ymax=30.0; // Maximum y value
  
  /* Parameters for the Gaussian initial conditions */
  const float x0=3.0;                    // Centre(x)
  const float y0=15.0;                    // Centre(y)
  const float sigmax=1.0;               // Width(x)
  const float sigmay=5.0;               // Width(y)
  const float sigmax2 = sigmax * sigmax; // Width(x) squared
  const float sigmay2 = sigmay * sigmay; // Width(y) squared

  /* Boundary conditions */
  const float bval_left=0.0;    // Left boudnary value
  const float bval_right=0.0;   // Right boundary value
  const float bval_lower=0.0;   // Lower boundary
  const float bval_upper=0.0;   // Upper bounary
  /* Constants for the velocity profile */
  const float u_star = 0.2; // Friction velocity (m/s)
  const float z0 = 1.0;     // Roughness length (m)
  const float kappa = 0.41; // von Kármán constant
  
  /* Time stepping parameters */
  const float CFL=0.9;   // CFL number 
  const int nsteps=800; // Number of time steps

  /* Velocity */
  /* const float velx=1.0; // Velocity in x direction */
  // Calculate maximum horizontal velocity at ymax
  float max_velx = (u_star / kappa) * log(ymax / z0);
  const float vely=0.0; // Velocity in y direction
  
  /* Arrays to store variables. These have NX+2 elements
     to allow boundary values to be stored at both ends */
  float x[NX+2];          // x-axis values
  float y[NX+2];          // y-axis values
  float u[NX+2][NY+2];    // Array of u values
  float dudt[NX+2][NY+2]; // Rate of change of u

  /* float x2;   // x squared (used to calculate iniital conditions) */
  /* float y2;   // y squared (used to calculate iniital conditions) */
  
  /* Calculate distance between points */
  float dx = (xmax-xmin) / ( (float) NX);
  float dy = (ymax-ymin) / ( (float) NY);
  
  /* Calculate time step using the CFL condition */
  /* The fabs function gives the absolute value in case the velocity is -ve */
  float dt = CFL / ( (fabs(max_velx) / dx) + (fabs(vely) / dy) );

 
  /*** Report information about the calculation ***/
  printf("Grid spacing dx     = %g\n", dx);
  printf("Grid spacing dy     = %g\n", dy);
  printf("CFL number          = %g\n", CFL);
  printf("Time step           = %g\n", dt);
  printf("No. of time steps   = %d\n", nsteps);
  printf("End time            = %g\n", dt*(float) nsteps);
  //printf("Distance advected x = %g\n", velx*dt*(float) nsteps);
  printf("Max horizontal velocity = %g\n", max_velx);
  printf("Distance advected y = %g\n", vely*dt*(float) nsteps);

  /*** Place x points in the middle of the cell ***/
  /*** LOOP 1 and LOOP 2 are initializing the x and y arrays respectively. 
    These are independent operations and can be parallelized. ***/
  /* LOOP 1 */
  #pragma omp parallel for
  for (int i=0; i<NX+2; i++){
    x[i] = ( (float) i - 0.5) * dx;
  }

  /*** Place y points in the middle of the cell ***/
  /* LOOP 2 */
  #pragma omp parallel for
  for (int j=0; j<NY+2; j++){
    y[j] = ( (float) j - 0.5) * dy;
  }

  /*** Set up Gaussian initial conditions ***/
  /*** LOOP 3: This nested loop calculates initial conditions independently for each grid point, so it can be parallelized.
    However, care must be taken to avoid race conditions. ***/
  /* LOOP 3 */
  #pragma omp parallel for collapse(2)
  for (int i=0; i<NX+2; i++){
    for (int j=0; j<NY+2; j++){
      float x2 = (x[i]-x0) * (x[i]-x0);
      float y2 = (y[j]-y0) * (y[j]-y0);
      u[i][j] = exp( -1.0 * ( (x2/(2.0*sigmax2)) + (y2/(2.0*sigmay2)) ) );
    }
  }

  /*** Write array of initial u values out to file ***/
  FILE *initialfile;
  initialfile = fopen("initial.dat", "w");
  /* LOOP 4 */
  /*** LOOP 4: File I/O should not be parallelized because the order of writing to the file must be preserved and concurrent writes could lead to a race condition. ***/
  for (int i=0; i<NX+2; i++){
    for (int j=0; j<NY+2; j++){
      fprintf(initialfile, "%g %g %g\n", x[i], y[j], u[i][j]);
    }
  }
  fclose(initialfile);
  
  /*** Update solution by looping over time steps ***/
  /* LOOP 5 */
  /*** LOOP 5: This is the time-stepping loop which inherently cannot be parallelized because each step depends on the previous one. ***/
  for (int m=0; m<nsteps; m++){
    
    /*New Loop made for task 2.3*/
    // Calculate horizontal velocity based on height y
    #pragma omp parallel for
    for (int j=1; j<NY+1; j++) {
        float velx_j;
        if (y[j] > z0) {
            velx_j = (u_star / kappa) * log(y[j] / z0);
        } else {
            velx_j = 0.0;
        }
    }
    /*** Apply boundary conditions at u[0][:] and u[NX+1][:] ***/
    /* LOOP 6 */
    /*** LOOP 6 and LOOP 7 apply boundary conditions. These are usually independent operations and can be parallelized. ***/
    #pragma omp parallel for
    for (int j=0; j<NY+2; j++){
      u[0][j]    = bval_left;
      u[NX+1][j] = bval_right;
    }

    /*** Apply boundary conditions at u[:][0] and u[:][NY+1] ***/
    /* LOOP 7 */
    #pragma omp parallel for
    for (int i=0; i<NX+2; i++){
      u[i][0]    = bval_lower;
      u[i][NY+1] = bval_upper;
    }
    
    /*** Calculate rate of change of u using leftward difference ***/
    /* Loop over points in the domain but not boundary values */
    /*** LOOP 8 calculates the rate of change dudt and is independent across iterations, making it suitable for 
    parallelization. However, care must be taken to ensure that the reading of u does not cause any race condition ***/
    /* LOOP 8 */
    #pragma omp parallel for collapse(2)
    for (int i=1; i<NX+1; i++){
        for (int j=1; j<NY+1; j++){
            float velx_j;
            if (y[j] > z0) {
                velx_j = (u_star / kappa) * log(y[j] / z0);
            } else {
                velx_j = 0.0;
            }
            dudt[i][j] = -velx_j * (u[i][j] - u[i-1][j]) / dx - vely * (u[i][j] - u[i][j-1]) / dy;
        }
    }

    
    /*** Update u from t to t+dt ***/
    /* Loop over points in the domain but not boundary values */
    /* LOOP 9 updates u from t to t+dt. As each grid point is updated independently, this loop can be parallelized similarly to LOOP 8. */
    /* LOOP 9 */
    #pragma omp parallel for collapse(2)
    for	(int i=1; i<NX+1; i++){
      for (int j=1; j<NY+1; j++){
	      u[i][j] = u[i][j] + dudt[i][j] * dt;
      }
    }
    
  }
   // time loop

  // Define an array to hold the vertically averaged values of u for each x position
  float u_avg[NX]; 

  // Loop over all x positions, excluding the boundary points
  for (int i = 1; i < NX + 1; i++) {
      float sum = 0; // Initialize a sum for each x position
 
      // Loop over all y positions for the current x, excluding the boundary points
      for (int j = 1; j < NY + 1; j++) {
          sum += u[i][j]; // Sum the values of u for the current x over all y
      }
      // Calculate the average by dividing the sum by the number of interior y points
      u_avg[i - 1] = sum / (NY - 2); // Average, excluding boundary values
  }

  FILE *avgfile = fopen("average.dat", "w");
  // Write the x position and the corresponding averaged value of u to the file
  for (int i = 0; i < NX; i++) {
      fprintf(avgfile, "%g %g\n", x[i + 1], u_avg[i]);     // Note: x[i + 1] is used to correctly match the x-axis values with the averaged u values
  }
  fclose(avgfile);
  
  /*** Write array of final u values out to file ***/
  FILE *finalfile;
  finalfile = fopen("final.dat", "w");
  /* LOOP 10 */
  /* LOOP 10 involves file I/O operations (writing to final.dat), similar to LOOP 4, and should not be parallelized.*/
  for (int i=0; i<NX+2; i++){
    for (int j=0; j<NY+2; j++){
      fprintf(finalfile, "%g %g %g\n", x[i], y[j], u[i][j]);
    }
  }
  fclose(finalfile);
  return 0;
}

/* End of file ******************************************************/