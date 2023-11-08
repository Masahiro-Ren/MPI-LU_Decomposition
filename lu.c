#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <mpi.h>

#define N 5760
#define NPROCS 576

#define MATRIX 1

#define EPS 2.220446e-16

double A[N][N];
double b[N];
double x[N];
double c[N];

int myid, numprocs;

void MyLUsolve(double A[N][N], double b[N], double x[N], int n);

int main(int argc, char *argv[])
{

    double t0, t1, t2, t_w;
    double dc_inv, d_mflops, dtemp, dtemp2, dtemp_t;

    int ierr;
    int i, j;
    int ii;
    int ib;

    ierr = MPI_Init(&argc, &argv);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    /* matrix generation --------------------------*/
    if (MATRIX == 1)
    {
        for (j = 0; j < N; j++)
        {
            ii = 0;
            for (i = j; i < N; i++)
            {
                A[j][i] = (N - j) - ii;
                A[i][j] = A[j][i];
                ii++;
            }
        }
    }
    else
    {
        srand(1);
        dc_inv = 1.0 / (double)RAND_MAX;
        for (j = 0; j < N; j++)
        {
            for (i = 0; i < N; i++)
            {
                A[j][i] = rand() * dc_inv;
            }
        }
    } 
    /* end of matrix generation -------------------------- */

    /* set vector b  -------------------------- */
    for (i = 0; i < N; i++)
    {
        b[i] = 0.0;
        for (j = 0; j < N; j++)
        {
            b[i] += A[i][j];
        }
    }
    /* ----------------------------------------------------- */

    // if (myid == 0) {                                                                                                                                                                                                                                                                         for(j=0; j<N; j++) {
    //     for(j = 0; j < N; j++)
    //     {
    //         for (i = 0; i < N; i++)
    //         {
    //             printf("%lf, ", A[j][i]);
    //         }
    //         printf("\n");
    //     }
    //     for(i = 0; i < N; i++)
    //         printf("%lf, ", b[i]);
    //     printf("\n");
    //     }
    // }
    // exit(0);

    /* Start of LU routine ----------------------------*/
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime();

    MyLUsolve(A, b, x, N);

    ierr = MPI_Barrier(MPI_COMM_WORLD);
    t2 = MPI_Wtime();
    t0 = t2 - t1;
    ierr = MPI_Reduce(&t0, &t_w, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    /* End of LU routine --------------------------- */

    if (myid == 0)
    {
        printf("N  = %d \n", N);
        printf("LU solve time  = %lf [sec.] \n", t_w);

        d_mflops = 2.0 / 3.0 * (double)N * (double)N * (double)N;
        d_mflops += 7.0 / 2.0 * (double)N * (double)N;
        d_mflops += 4.0 / 3.0 * (double)N;
        d_mflops = d_mflops / t_w;
        d_mflops = d_mflops * 1.0e-6;
        printf(" %lf [MFLOPS] \n", d_mflops);
    }

    /* Verification routine ----------------- */
    ib = N / NPROCS;
    dtemp_t = 0.0;
    for (j = myid * ib; j < (myid + 1) * ib; j++)
    {
        dtemp2 = x[j] - 1.0;
        dtemp_t += dtemp2 * dtemp2;
    }
    dtemp_t = sqrt(dtemp_t);
    /* -------------------------------------- */

    MPI_Reduce(&dtemp_t, &dtemp, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    /* Do not modify follows. -------- */
    if (myid == 0)
    {
        if (MATRIX == 1)
            dtemp2 = (double)N * (double)N * (double)N;
        else
            dtemp2 = (double)N * (double)N;
        dtemp_t = EPS * (double)N * dtemp2;
        printf("Pass value: %e \n", dtemp_t);
        printf("Calculated value: %e \n", dtemp);
        if (dtemp > dtemp_t)
        {
            printf("Error! Test is falled. \n");
            exit(1);
        }
        printf(" OK! Test is passed. \n");
    }
    /* ----------------------------------------- */

    ierr = MPI_Finalize();

    exit(0);
}

void MyLUsolve(double A[N][N], double b[N], double x[N], int n)
{
    int i, j, k;
    int kstart, kend;
    int ib;
    int ipivotPE, idiagPE;
    double dtemp;
    double buf[N];

    MPI_Status istatus;

    /* Calculate range */
    ib = n / numprocs; 
    kstart = myid * ib;
    kend = (myid + 1) * ib;

    /* LU decomposition ----------------------------------------------------------------------- */
    for (k = 0; k < kend; k++)
    {
        
        ipivotPE = k / ib;
        /* process Matrix L */
        if(myid == ipivotPE)
        {
            dtemp = 1.0 / A[k][k];
            for (i = k + 1; i < n; i++)
            {
                A[i][k] = A[i][k] * dtemp;
                buf[i] = A[i][k];
            }
            kstart = k + 1;
            /* send pivot vector to other process */
            for(int isendPE = myid + 1; isendPE < numprocs; isendPE++)
            {
                MPI_Send(&buf[k + 1], n - k - 1, MPI_DOUBLE, isendPE, ipivotPE, MPI_COMM_WORLD);
            }
        }
        else
        {
            /* receive pivot vector from other process*/
            MPI_Recv(&buf[k + 1], n - k - 1, MPI_DOUBLE, ipivotPE, ipivotPE, MPI_COMM_WORLD, &istatus);
        }
        /* Process Matix U */
        for (j = k + 1; j < n; j++)
        {
            dtemp = buf[j];
            for (i = kstart; i < kend; i++)
            {
                A[j][i] = A[j][i] - A[k][i] * dtemp;
            }
        }
    }

    // printf("pe %d : \n", myid);
    // for(i = 0; i < n; i++)
    // {
    //     for(j = myid * ib; j < (myid + 1) * ib; j++)
    //     {
    //         printf("%lf ", A[i][j]);
    //     }
    //     printf("\n");
    // }

    /* LU decomposition end ----------------------------------------------------------------------- */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Forward substitution ------------------------------------------------------------------------*/
    /* Calculate range */
    kstart = myid * ib;
    kend = (myid + 1) * ib;
    for(k = 0; k < n; k++)
        c[k] = 0.0;
    for(k = 0; k < n; k += ib)
    {
        if( k >= kstart) 
        {
            idiagPE = k / ib;        
            if(myid != 0)
            {
                MPI_Recv(&c[k], ib, MPI_DOUBLE, myid - 1, k, MPI_COMM_WORLD, &istatus);
            }
            if(myid == idiagPE)
            {
                for(int kk = 0; kk < ib; kk++)
                {
                    c[k + kk] = b[k + kk] + c[k + kk];
                    for(j = kstart; j < kstart + kk; j++)
                    {
                        c[k + kk] -= A[k + kk][j] * c[j];
                    }
                }
            }
            else
            {
                for (int kk = 0; kk < ib; kk++)
                {
                    for (j = kstart; j < kend; j++)
                    {
                        c[k + kk] -= A[k + kk][j] * c[j];
                    }
                }

                if(myid != numprocs - 1)
                {
                    MPI_Send(&c[k], ib, MPI_DOUBLE, myid + 1, k, MPI_COMM_WORLD);
                }
            }
        }
    }

    // print C
    // printf("Process %d : ", myid);
    // for(int m = kstart; m < kend; m++)
    //     printf("%lf ", c[m]);
    // printf("\n");

    /* Forward substitution end ------------------------------------------------------------------------*/

    /* Backward subsititution --------------------------------------------------------------------------*/
    for(k = 0; k < n; k++)
        x[k] = 0.0;
    
    for(k = n - 1; k >= 0; k -= ib)
    {
        if (k < kend)
        {
            idiagPE = k / ib;
            if (myid != numprocs - 1)
            {
                MPI_Recv(&x[k - ib + 1], ib, MPI_DOUBLE, myid + 1, k, MPI_COMM_WORLD, &istatus);
            }
            if (myid == idiagPE)
            {
                for (int kk = 0; kk < ib; kk++)
                {
                    x[k - kk] = c[k - kk] + x[k - kk];
                    for (j = k - kk + 1; j < kend; j++)
                    {
                        x[k - kk] -= A[k - kk][j] * x[j];
                    }
                    x[k - kk] = x[k - kk] / A[k - kk][k - kk];
                }
            }
            else
            {
                for (int kk = 0; kk < ib; kk++)
                {
                    for (j = kstart; j < kend; j++)
                    {
                        x[k - kk] -= A[k - kk][j] * x[j];
                    }
                }
                if (myid != 0)
                {
                    MPI_Send(&x[k - ib + 1], ib, MPI_DOUBLE, myid - 1, k, MPI_COMM_WORLD);
                }
            }
        }
    }
    // printf("Process %d : ", myid);
    // for(int m = kstart; m < kend; m++)
    //     printf("%lf ", x[m]);
    // printf("\n");

    /* Backward subsititution end --------------------------------------------------------------------------*/
}

// void MyLUsolve(double A[N][N], double b[N], double x[N], int n)
// {
//     int i, j, k;
//     int kstart, kend;
//     int ib;
//     int ipivotPE, idiagPE;
//     double dtemp;
//     double buf[N];

//     MPI_Status istatus;

//     /* Calculate range */
//     ib = n / numprocs; 
//     kstart = myid * ib;
//     kend = (myid + 1) * ib;
//     /* LU decomposition ---------------------- */
//     for (k = 0; k < kend; k++)
//     {
        
//         ipivotPE = k / ib;
//         /* process Matrix L */
//         if(myid == ipivotPE)
//         {
//             dtemp = 1.0 / A[k][k];
//             for (i = k + 1; i < n; i++)
//             {
//                 A[i][k] = A[i][k] * dtemp;
//                 buf[i] = A[i][k];
//             }
//             kstart = k + 1;
//             /* send pivot vector to other process */
//             for(int isendPE = myid + 1; isendPE < numprocs; isendPE++)
//             {
//                 MPI_Send(&buf, N, MPI_DOUBLE, isendPE, ipivotPE, MPI_COMM_WORLD);
//             }
//         }
//         else
//         {
//             /* receive pivot vector from other process*/
//             MPI_Recv(&buf, N, MPI_DOUBLE, ipivotPE, ipivotPE, MPI_COMM_WORLD, &istatus);
//         }
//         /* Process Matix U */
//         for (j = k + 1; j < n; j++)
//         {
//             dtemp = buf[j];
//             for (i = kstart; i < kend; i++)
//             {
//                 A[j][i] = A[j][i] - A[k][i] * dtemp;
//             }
//         }
//     }
//     /* --------------------------------------- */
//     MPI_Barrier(MPI_COMM_WORLD);

//     /* Forward substitution ------------------ */
//     for (k = 0; k < n; k++)
//     {
//         c[k] = b[k];
//         for (j = 0; j < k; j++)
//         {
//             c[k] -= A[k][j] * c[j];
//         }
//     }
//     /* --------------------------------------- */

//     /* Backward substitution ------------------ */
//     x[n - 1] = c[n - 1] / A[n - 1][n - 1];
//     for (k = n - 2; k >= 0; k--)
//     {
//         x[k] = c[k];
//         for (j = k + 1; j < n; j++)
//         {
//             x[k] -= A[k][j] * x[j];
//         }
//         x[k] = x[k] / A[k][k];
//     }
//     /* --------------------------------------- */
// }
