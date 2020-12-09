#include <omp.h>
#include <math.h>
#include <iostream>
#include <mpi.h>

using namespace std;


// serial function
void multiply(int n, int* a[], int* b[], int* res[])
{
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			res[i][j] = 0;
			for(int k = 0; k < n; k++)
				res[i][j] += a[i][k] * b[k][j];
		}
	}
}

void multParallel_1(int n)
{

	int rank, world;
	
	// intializing data
	int a[n][n], b[n][n], res[n][n];
	

	#pragma omp parallel for
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			a[i][j] = 1;
			b[i][j] = 1;
		}
	}


	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world);

	double start = MPI_Wtime();
	// intializing buffers
	int rows = n / world;
	int recBuf_a[rows][n], res_c[rows][n];

	// sending rows and broadcasting the second matrix
	MPI_Scatter(a, rows * n, MPI_INT, &recBuf_a, rows * n, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(b, n*n, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);

	// calculating rows * n part of the result matrix
	for(int k = 0; k < rows; k++)
	{
		for(int i = 0; i < n; i++)
		{
			int sum = 0;
			for(int j = 0; j < n; j++)
			{
				sum += recBuf_a[k][i] * b[j][i];
			}
			res_c[k][i] = sum;
		}
	}
	
	// collecting and displaying results
	MPI_Gather(res_c, n*rows, MPI_INT, res, n*rows, MPI_INT, 0, MPI_COMM_WORLD);
	
	MPI_Barrier(MPI_COMM_WORLD);
	if(rank == 0)
		cout << "Time for parallel 1: " << MPI_Wtime() - start << endl;

	//MPI_Barrier(MPI_COMM_WORLD);
	/**if(rank == 0)
	{
		cout << n << endl;
		cout << "finished gather" << endl;
		for(int i = 0; i < n; i++)
		{
			for(int j = 0; j < n; j++)
				cout << res[i][j] << " ";
			cout << endl;
		}
	}**/
}


void multParallel_2(int n)
{

	int a[n][n], b[n][n], res[n][n];


	int rank, world;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world);

	if (rank == 0)
	for(int i = 0; i < n; i++)
	{
		for(int j = 0; j < n; j++)
		{
			a[i][j] = 1;
			b[i][j] = 1;
		}
	}
	
	MPI_Barrier(MPI_COMM_WORLD);

	double start = MPI_Wtime();

	int rows = n / world;
	int recBuf_b[n][rows], res_c[n][rows];

	MPI_Bcast(a, n*n, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Datatype coltype;

	MPI_Type_vector(n, 1, n, MPI_INT, &coltype);
	MPI_Type_commit(&coltype);

	MPI_Datatype col_recv;

	MPI_Type_vector(n, 1, rows, MPI_INT, &col_recv);
	MPI_Type_commit(&col_recv);

	if(rank == 0)
	{
		for(int i = 0; i < world; i++)
		{
			for(int j = 0; j < rows; j++)
			{
				MPI_Send(&b[0][j + i * rows * n], 1, coltype, i, 0, MPI_COMM_WORLD);
			}
		}
	}

	//else
	for(int i = 0; i < rows; i++)
	{
		MPI_Recv(&recBuf_b[0][i], 1, col_recv, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	}
	MPI_Barrier(MPI_COMM_WORLD);
		

	for(int k = 0; k < n; k++)
	{
		for(int i = 0; i < rows; i++)
		{
			int sum = 0;
			for(int j = 0; j < n; j++)
			{
				sum += a[k][j] * recBuf_b[j][i];
			}
			res_c[k][i] = sum;
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Gather(res_c, n*rows, MPI_INT, res, n*rows, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);

	if(rank == 0)
		cout << "Time for parallel 2: " << MPI_Wtime() - start << endl;

	/**if(rank == 0)
	{
		cout << "finished gather" << endl;
		for(int i = 0; i < n; i++)
		{
			for(int j = 0; j < n; j++)
				cout << res[i][j] << " ";
			cout << endl;
		}
	}**/
}

int main(int argc, char *argv[])
{

	const int n = 1296;

	int** a = new int*[n];
	int** b = new int*[n];
	int** res = new int*[n];
	for(int i = 0; i < n; i++)
	{
		a[i] = new int[n];
		b[i] = new int[n];
		res[i] = new int [n];
		for(int j = 0; j < n; j++)
		{
			a[i][j] = 1;
			b[i][j] = 1;
		}
	}
	double start = MPI_Wtime();
	multiply(n, a, b, res);
	cout << "Time for serial: " << MPI_Wtime() - start << endl;
	

	MPI_Init(NULL, NULL);

	multParallel_1(n);
	multParallel_2(n);

	MPI_Finalize();

	return 0;
}