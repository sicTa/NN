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
                       w[i] = (double)(rand()%10) / 10;
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
       
       
       void ispisi()
       {
            for(int i = 0; i < n; i++)
                    printf("%.2llf   ", w[i]);
                    printf("\n");
        }

};



Perceptron hidden[5] = 
{
     Perceptron(3),
     Perceptron(3),
     Perceptron(3),
     Perceptron(3),
     Perceptron(3)
};
Perceptron output(6);


void uci(vector<vector<double> > x, vector<int> y)
{
     for(int brojac = 0; brojac < 900; brojac++)
     {
     
        for(int p = 0; p < x.size(); p++)
             {
                     vector<double> z;
                     
                     //propagacija unapred
                     //izracunavanje rezultata skrivenih neurona
                     for(int i = 0; i < 5; i++)
                         z.push_back(hidden[i].exp_sigmoid(x[p]));
                    
                                            
                     //izracunavanje rezultata izlaza
                     double y_izlaz = output.linear_sigmoid(z);
                     
                     
                     //izracunavanje gradijenta greske
                     double delta_y = 2 * (y_izlaz - (double)y[p]);
             
                     //modifikovanje svih tezina
                     
                     //modifikovanje vektora tezina izlaza
                     for(int i = 0; i < output.w.size(); i++)
                         {
                             double trenutno = ITA * delta_y * z[i];
                             output.w[i] = output.w[i] - trenutno;
                         }
                         
                     //modifikovanje vektora tezina skrivenih neurona
                     for(int i = 0; i < 5; i++)
                         for(int j = 0; j < hidden[i].w.size(); j++)
                             {
                                 double trenutno = x[p][j] * ITA * delta_y * z[i] * (1.0 - z[i]) * output.w[i];
                                 hidden[i].w[j] = hidden[i].w[j] - trenutno;
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
            

            
            double x1p = (double)x1 + (double)(rand()%10) / 100;
            double x2p = (double)x2 + (double)(rand()%10) / 100;

            vector<double> pom = vector<double>(3);
            pom[0] = 1.0;
            pom[1] = x1p;
            pom[2] = x2p;
            x.push_back(pom);
            
            
            y.push_back(yp);
            
    }
     output.ispisi();
    uci(x, y);
    
    
    
    //racunanje rezultata

     
     int tacnih = 0;
     for(int i = 0; i < 50; i++)
     {
         vector<double> test;
         //za bias
         test.push_back(1.0);
     
     
     
         int x1 = rand()%2;
         int x2 = rand()%2;
         test.push_back(x1);
         test.push_back(x2);
     
         vector<double> z;
         for(int i = 0; i < 5; i++)
         {
             double a = hidden[i].exp_sigmoid(test);
             z.push_back(a);
         }
         
         
         double y_output = output.linear_sigmoid(z);
         
         if(y_output > 0.5 && x1 != x2) tacnih++;
         else if(y_output <= 0.5 && x1 == x2)tacnih++;
         
         printf("Ulaz: %d %d  Izlaz: %.3llf\n", x1, x2, y_output);         
     }
     
     printf("Tacno izracunato %d od 50\n", tacnih);
    system("PAUSE");
    return EXIT_SUCCESS;
}
