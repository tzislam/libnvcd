
/*
 *  Parallel Computing Assignment 1: Matrix multiplication
 *
 */

#include <algorithm>
#include <omp.h>
#include <ctime>
#include <deque>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <stack>
#include <string>
#include <utility>
#include <vector>
#include <assert.h>

using namespace std;

#define pks printf("Case %d: ",++ks);
#define mx 1002

int a[mx][mx];
int b[mx][mx];
int c[mx][mx];
int d[mx][mx];

void generate_matrix(int n)
{
	int i,j;
	for(i=0;i<n;i++)
	{
		for(j=0;j<n;j++)
		{
			a[i][j] = rand() % 100;
			b[i][j] = rand() % 100;
		}
	}
}

static inline void assert2(int cond)
{
	if (!cond)
	{
		puts("ERROR");
		exit(1);
	}
}

void check(int n)
{
	int i,j;
	for(i=0;i<n;i++)
	{
		for(j=0;j<n;j++)
			assert2(c[i][j]==d[i][j]);
	}
}

void matrix_mult_serial(int n)
{
	int i,j,k;
	double st=omp_get_wtime();
	for(i=0;i<n;i++)
	{
		for(j=0;j<n;j++)
		{
			for(k=0;k<n;k++)
			{
				c[i][j]+=a[i][k]*b[k][j];
			}
		}
	}
	double en=omp_get_wtime();
	printf("(%lf,",en-st);
}

void matrix_mult_parallel(int n)
{
	int i, j, k;
	double st = omp_get_wtime();
	for (i = 0; i < n; ++i)
	{
		for (j = 0; j < n; ++j)
		{
			int u = 0;
#pragma omp parallel for reduction(+:u)
			for (k = 0; k < n; ++k)
			{
				u += a[i][k] * b[k][j]; 
			}
			d[i][j] = u;
		}
	}

	double en = omp_get_wtime();
	printf("%lf),\n", en - st);
}

int main(int argc, char *argv[]) {
	//int n=500
	int n;
	if (argc < 2){
		printf("Usage: %s matrix_dimension\n", argv[0]);
		return 0;
	}
	n = atoi(argv[1]);
	generate_matrix(n);
	matrix_mult_serial(n);
	matrix_mult_parallel(n);
	return 0;
}
