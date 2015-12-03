/*
 Petar 'PetarV' Velickovic
 Algorithm: Logistic Perceptron
*/

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include <list>
#include <string>
#include <algorithm>
#include <queue>
#include <stack>
#include <set>
#include <map>
#include <complex>

#define EPS 1e-6
#define ETA 1e-3

using namespace std;
typedef long long lld;

class Perceptron
{
private:
    int n;
    vector<double> w;
    
    double scalar_product(vector<double> a, vector<double> b)
    {
        assert(a.size() == b.size());
        double ret = 0.0;
        for (int i=0;i<a.size();i++)
        {
            ret += a[i] * b[i];
        }
        return ret;
    }
    
public:
    Perceptron(int n) : n(n)
    {
        w = vector<double>(n);
        w[0] = 0; w[1] = 0;
    }
    
    double val(vector<double> x)
    {
        return 1.0 / (1.0 + exp(-scalar_product(w, x)));
    }
    
    void train(vector<vector<double> > x, vector<int> y)
    {
        assert(x.size() == y.size());
        vector<double> delta(n, 0.0);
        double diff = 0.0;
        do
        {
            diff = 0.0;
            for (int i=0;i<x.size();i++)
            {
                double lst = val(x[i]);
                for (int j=0;j<n;j++)
                {
                    double curr = ETA * (y[i] - lst) * lst * (1.0 - lst) * x[i][j];
                    diff += curr * curr;
                    w[j] += curr;
                }
            }
            printf("w = ");
            for (int j=0;j<n;j++)
            {
                printf("%.3lf ", w[j]);
            }
            printf("\n");
            //scanf("%*d");
        } while (diff > EPS);
    }
};

int main()
{
    srand(time(NULL));
    
    int t = 1000;
    
    vector<vector<double> > trn;
    vector<int> vals;
    for (int i=0;i<t;i++)
    {
        int x = rand() % 5 + 1;
        int y = rand() % 5 + 1;
        while (y == x) y = rand() % 5 + 1;
        vector<double> set;
        set.push_back(x); set.push_back(y);
        trn.push_back(set);
        if (x > y) vals.push_back(1);
        else vals.push_back(0);
    }
    Perceptron p(2);
    p.train(trn, vals);
    
    int correct = 0;
    for (int i=0;i<t;i++)
    {
        double x = p.val(trn[i]);
        if (x > 0.5 && vals[i] == 1) correct++;
        else if (x < 0.5 && vals[i] == 0) correct++;
    }
    cout << "correct = " << correct << endl;
    
    return 0;
}
