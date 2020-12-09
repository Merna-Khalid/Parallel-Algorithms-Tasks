#include <omp.h>
//#include <pthread.h>
#include <math.h>
#include <iostream>

using namespace std;

double f(double x)
{
	return (1 / (x * x)) * (pow(sin(1/x), 2));
}

double integral(int n, double a, double b)
{
	
	double part = (b - a) / double(n);
	double x_i = a;
	double sum = 0;
	sum += f(x_i) / 2.0;
	for(int i = 1; i < n; i++)
	{
		x_i = a + part * i;
		sum += f(x_i);
	}
	x_i = a + part * n;
	return part * (sum + f(x_i) / 2.0);
}

double integralAtomic(int n, double a, double b)
{
	//int max_threads = omp_get_num_threads();
	double part = (b - a) / double(n);
	double x_i = a;
	double sum = 0;
	sum += f(x_i) / 2.0;
	int i = 1;
	#pragma omp parallel 
	{
		#pragma omp for private(i)
		for(i = 1; i < n; i++)
		{
			#pragma omp atomic update
			sum += f(a + part * i);
		}
	}
	x_i = a + part * n;
	return part * (sum + f(x_i) / 2.0);
}

double integralCritical(int n, double a, double b)
{
	double part = (b - a) / double(n);
	double x_i = a;
	double sum = 0;
	sum += f(x_i) / 2.0;
	#pragma omp parallel 
	{
		#pragma omp for
		for(int i = 1; i < n; i++)
		{
			#pragma omp critical
			sum += f(a + part * i);
		}
	}
	x_i = a + part * n;
	return part * (sum + f(x_i) / 2.0);
}

double integralLock(int n, double a, double b)
{
	double part = (b - a) / double(n);
	double x_i = a;
	double sum = 0;
	sum += f(x_i) / 2.0;

	omp_lock_t writelock;
	omp_init_lock(&writelock);

	int i = 1;
	#pragma omp parallel 
	{
		#pragma omp for private(i)
		for(i = 1; i < n; i++)
		{
			omp_set_lock(&writelock);
			sum += f(a + part * i);
			omp_unset_lock(&writelock);
		}
	}
	omp_destroy_lock(&writelock);
	x_i = a + part * n;
	return part * (sum + f(x_i) / 2.0);
}

double integralRedc(int n, double a, double b)
{
	double part = (b - a) / double(n);
	double x_i = a;
	double sum = 0;
	sum += f(x_i) / 2.0;

	#pragma omp parallel for reduction(+:sum)
	for(int i = 1; i < n; i++)
	{
		sum += f(a + part * i);
	}
	
	x_i = a + part * n;
	return part * (sum + f(x_i) / 2.0);
}

double n_loop(int max_n, double e, double a, double b, double (*f)(int, double, double))
{
	double j_n1, j_n2;
	j_n1 = f(1, a, b);
	for(int i = 2; i < max_n; i++)
	{
		j_n2 = integral(i, a, b);
		//if(abs(j_n2 - j_n1) <= e * j_n2)
		//	break;
		j_n1 = j_n2;
	}
	return j_n2;
}

int main()
{
	long long n = 1000;
	double a = 10, b = 100;
	double res;
	double start;
	//res = n_loop(n, 0.0000001, a, b, integral);


	int max_threads = omp_get_max_threads();
	cout << "max threads : " << max_threads << endl; 

	omp_set_dynamic(0);
	omp_set_num_threads(12);

	int x = 1;
	#pragma omp parallel 
	{
		#pragma omp for
		for(int i = 0; i < 12; i++)
		{
			x = 1;
		}
	}

	cout << "Serial" << endl;
	start = omp_get_wtime();
	res = integral(n, a, b);
	cout << abs(omp_get_wtime() - start) << endl;
	cout << res << endl;
	cout << endl;

	cout << "Atomic" << endl;
	start = omp_get_wtime();
	res = integralAtomic(n, a, b);
	cout << abs(omp_get_wtime() - start) << endl;
	cout << res << endl;
	cout << endl;

	cout << "Critical" << endl;
	start = omp_get_wtime();
	res = integralCritical(n, a, b);
	cout << abs(omp_get_wtime() - start) << endl;
	cout << res << endl;
	cout << endl;

	cout << "Lock" << endl;
	start = omp_get_wtime();
	res = integralLock(n, a, b);
	cout << abs(omp_get_wtime() - start) << endl;
	cout << res << endl;
	cout << endl;

	cout << "Reduction" << endl;
	start = omp_get_wtime();
	res = integralRedc(n, a, b);
	cout << abs(omp_get_wtime() - start) << endl;
	cout << res << endl;
	cout << endl;

	return 0;
}