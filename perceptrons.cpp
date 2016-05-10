#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <algorithm>

#define ITA 1e-2
using namespace std;

class Perceptron
{
private:
        //broj elementa vektora w koji idu u neuron
        int n;
        double skalarni_proizvod(vector<double> a, vector<double> b)
        {
               double suma = 0.0;
               for(int i = 0; i < a.size(); i++)
                       suma+=a[i] * b[i];
               return suma;      
        }
        
        
public:
       //rezultat izracnavanja
       //A je skalarni proizvod w i x(vektor aparametara)
       //Z je rezultat sigmoidne funkcije
       vector<double> w;
       Perceptron(int n):n(n)
       {
               w = vector<double>(n);
               for(int i = 0; i < n; i++)
                       w[i] = 0.0;
       }
       
       //aktivaciona funkcija output neurona
       double linear_sigmoid(vector<double> x)
       {
              return  skalarni_proizvod(w, x);
       }
       
       //aktivaciona funkcija skrivenog neurona
       double exp_sigmoid(vector<double> x)
       {
              return 1.0 / (1.0 + exp(-skalarni_proizvod(w, x)));
       }

};



Perceptron hidden[5] = 
{
     Perceptron(2),
     Perceptron(2),
     Perceptron(2),
     Perceptron(2),
     Perceptron(2)
};
Perceptron output(2);


void uci(vector<vector<double> > x, vector<int> y)
{
     for(int brojac = 0; brojac < 900; brojac++)
     {
     
     
     double trenutno = 0.0;
     for(int br = 0; br < x.size(); br++)
     {
             //izracunavanje rezultata svih neurona
             //vektor koji cuva rezultate svih perceptrona
             vector<double> z =  vector<double>(5);
             for(int i = 0; i < 5; i++)
             {
                double a = hidden[i].exp_sigmoid(x[br]);
                z.push_back(a);
             }
             double y_output = output.linear_sigmoid(z);
             
             
             //racunanje ukupne greske
             double delta_y =y_output - y[br];
             for(int i = 0; i < 5; i++)
                     trenutno+=2 * z[i] * delta_y;
                     
             
             for(int j = 0; j < 5; j++)
                 for(int i = 0; i < x[0].size(); i++)
                         trenutno += 2 * x[br][i] * z[j] * delta_y * output.w[j];
                         
             
             //menjanje vrednost svih tezina
             for(int j = 0; j < 5; j++)
             {
                 output.w[j] += ITA * trenutno;
                 for(int i = 0; i < hidden[0].w.size(); i++)
                     hidden[j].w[i] += ITA * trenutno; 
             }
     }
     
     
     
     
     
     
     
     }
}

int main()
{
    srand(time(NULL));

    vector<vector<double> > x;
    vector<int> y;
    


    for(int i = 0; i < 40; i++)
    {
            int x1 = rand()%2;
            int x2 = rand()%2;
            int yp  = x1 ^ x2;
            

            
            double x1p = x1 + (rand()%100 - 50)/ 1000;
            double x2p = x2 + (rand()%100 - 50)/ 1000;
            
            vector<double> pom = vector<double>(2);
            pom.push_back(x1p);
            pom.push_back(x2p);
            x.push_back(pom);
            
            y.push_back(yp);
            
    }
    uci(x, y);
    
    
    
    //racunanje rezultata
     vector<double> test = vector<double>(2);
     test.push_back(1.0);
     test.push_back(1.0);
     vector<double> z =  vector<double>(5);
     for(int i = 0; i < 5; i++)
     {
         double a = hidden[i].exp_sigmoid(test);
         z.push_back(a);
     }
     double y_output = output.linear_sigmoid(z);
     printf("%lld", y_output);
    
    system("PAUSE");
    return EXIT_SUCCESS;
}
