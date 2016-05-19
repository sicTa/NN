#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <algorithm>

#define ITA 1e-7
using namespace std;

/*

fiksne stvari: dva izlazna identitet neurona + dva softmax neurona
cross-entropy loss
varijabilne stvari:
broj neurona, broj skrivenih slojeva (1/2 ~ shallow/deep), aktivaciona funkcija (sigmoid/ReLU)
*/


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
       
       //aktivaciona funkcija prvog dela output neurona
       double linear_sigmoid(vector<double> x)
       {
              return  skalarni_proizvod(w, x);
       }
       
       //aktivaciona funkcija drugog dela output neurona
       double softmax(vector<double> x, int j)
       {
              double sum = 0.0;
              for (int i = 0; i < x.size(); i++)
                   sum+=exp(x[i]);
              return exp(x[j])/sum;
       }
       
       //aktivaciona funkcija skrivenog neurona
       double ReLU(vector<double> x)
       {
              double proizvod = skalarni_proizvod(w, x);
              return (proizvod > 0) ? proizvod : 0;
       }
 
       //
            
       void ispisi()
       {
            for(int i = 0; i < n; i++)
                    printf("%.2llf   ", w[i]);
                    printf("\n");
       }

};

//kreacija neurona
vector<vector<Perceptron> > hidden;
vector<Perceptron> output;

/* NAPOMENA!
output.size() = 2 zato sto imamo dve klase koje trazimo
*/
void uci(vector<vector<double> > x, vector<vector<int> > y)
{
     for(int brojac = 0; brojac < 900; brojac++)
     {   
        for(int p = 0; p < x.size(); p++)
             {
                     vector<vector<double> > z;
                     vector<double>pom;
                     
                     //propagacija unapred
                     //izracunavanje rezultata skrivenih neurona
                     for(int i = 0; i < hidden.size(); i++)
                         {
                             //dodajemo 1 u pom zbog bias
                             pom.push_back(1.0);
                             for(int j = 0; j < hidden[i].size(); j++)
                                 {
                                     if(i == 0)
                                        pom.push_back(hidden[i][j].ReLU(x[p]));
                                     else
                                        pom.push_back(hidden[i][j].ReLU(z[i-1]));                                      
                                 }
                             z.push_back(pom);
                             pom.clear();
                         }
                     pom.clear(); 
                                                           
                     //izracunavanje rezultata izlaza
                     vector<double> izlaz;
                     vector<double> izlaz_linear; 
                                   
                         //izracunavanje linearnih izlaza iz neurona
                     for(int i = 0; i < output.size(); i++)
                         izlaz_linear.push_back(output[i].linear_sigmoid(z.back()));
                         //izracunavanje softmax funkcije za svaki izlazni neuron 
                     for(int i = 0; i < output.size(); i++)
                     {
                         izlaz.push_back(output[i].softmax(izlaz_linear, i));
                         
                     }
                      
                     //izracunavanje gradijenta greske
                     vector<double> delta_y;
                     for(int i = 0; i < output.size(); i++)
                         delta_y.push_back(y[p][i] - izlaz[i]);
                                             
                     //MODIFIKOVANJE SVIH TEZINA
                     
                     //modifikovanje vektora tezina izlaza
                     for(int i = 0; i < output.size(); i++)
                         for(int t = 0; t < output[i].w.size(); t++)
                             {
                                 double trenutno = ITA * z.back()[t] * delta_y[i];
                                 output[i].w[t] = output[i].w[t] - trenutno;
                             }
                             
                     //modifikacija vektora tezina skrivenih neurona
                     vector<vector<double> > delta_h;
                     pom.clear(); 
                     
                     int MAX_SLOJEVA = hidden.size() - 1;
                     for(int i = MAX_SLOJEVA; i >= 0; i--)
                         {
                             for(int j = 0; j < hidden[i].size(); j++)
                                 {
                                     int izvod;
                                     if(z[i][j] <= 0)izvod = 0;
                                     else izvod = 1;
                                     
                                     double suma = 0.0;
                                     if(i == MAX_SLOJEVA)
                                     {
                                         for(int k = 0; k < output.size(); k++)
                                             suma += delta_y[k] * output[k].w[j + 1];//doslo do izmene
                                     }
                                     else
                                     {
                                         for(int k = 0; k < hidden[i + 1].size(); k++)
                                             suma += delta_h[MAX_SLOJEVA - i - 1][k] * hidden[i + 1][k].w[j + 1];
                                     }
                                     
                                     double delta_j = suma * izvod;
                                     pom.push_back(delta_j);
                                     
                                     //modifikacija vektora tezina za neuron j u i-tom sloju
                                     //gledamo trenutni delta_j kao i rezultat tog neurona
                                     //ako se nalazimo na prvom skrivenom neuronu
                                     //kao podatke treba uzeti vrednosti ulaznih vektora
                                     //u suprotnom, koristiti rezultat racunanja prethodnih slojeva       
                                    if(i == 0)
                                       {
                                            for(int k = 0; k < hidden[i][j].w.size(); k++)
                                            {
                                                double trenutno = x[p][k] * ITA * delta_j; 
                                                hidden[i][j].w[k] = hidden[i][j].w[k] - trenutno;                           
                                            }
                                       }
                                    else
                                       {
                                            for(int k = 0; k < hidden[i][j].w.size(); k++)
                                            {
                                                double trenutno = ITA * delta_j * z[i - 1][k];
                                                hidden[i][j].w[k] = hidden[i][j].w[k] - trenutno;                        
                                            }
                                       }                                                                     
                                 }
                                 delta_h.push_back(pom);
                                 pom.clear();
                         }               
             }       
     }
}


void sracunaj(vector<double> x)
{
     //metoda za ispisivanje rezultat mreze  
     vector<vector<double> > z;
                     vector<double>pom;
                     
                     //propagacija unapred
                     //izracunavanje rezultata skrivenih neurona
                     for(int i = 0; i < hidden.size(); i++)
                         {
                             //dodajemo 1 u pom zbog bias
                             pom.push_back(1.0);
                             for(int j = 0; j < hidden[i].size(); j++)
                                 {
                                     if(i == 0)
                                        pom.push_back(hidden[i][j].ReLU(x));
                                     else
                                        pom.push_back(hidden[i][j].ReLU(z[i-1]));
                                 }
                             z.push_back(pom);
                             pom.clear();
                         }
                     pom.clear(); 
                                                           
                     //izracunavanje rezultata izlaza
                     vector<double> izlaz;
                     vector<double> izlaz_linear; 
                                   
                         //izracunavanje linearnih izlaza iz neurona
                     for(int i = 0; i < output.size(); i++)
                         izlaz_linear.push_back(output[i].linear_sigmoid(z.back()));
                         //izracunavanje softmax funkcije za svaki izlazni neuron 
                     for(int i = 0; i < output.size(); i++)
                         izlaz.push_back(output[i].softmax(izlaz_linear, i));  
                         
                         
                         
                     for(int i = 0; i < izlaz.size(); i++)
                         printf("Verovatnoca pripadanja klasi %d je: %.2llf\n", i, izlaz[i]); 
                     printf("********************************************\n");
}



int main()
{
    //kreiranje mreze i outputa
    //imamo 2 skrivena sloja
    //za sada, svaki sloj skrivene mreze imace 5 neurona
    //output sloj imace 2 neurona
    //vektor parametara imace 3 clama, prvi = 1
    //da bi se pravilno uracunao bias
    
    //ubacivanje u prvi sloj skrivenih neurona
    vector<Perceptron> pom;
    for(int i = 0; i < 5; i++)
        pom.push_back(Perceptron(3));
    hidden.push_back(pom);
    pom.clear();
    
    //ubacivanje u drugi sloj skrivenih neurona
    //napomena: vektor parametara imace 6 clanova
    //tj jedan vise od broja elemenata prethodnog sloja
    //da bi se uracunali svi rezultati prvog sloja i bias za 
    //svaki neuron
    for(int i = 0; i < 5; i++)
        pom.push_back(Perceptron(6));
    hidden.push_back(pom);
    pom.clear();
    
    
    //ubacivanje u sloj output neurona
    //napomena: vektor parametara imace 6 clanova
    //tj 1 vise od broja elemenata prethodnog sloja
    //output sloj imace 2 neurona
    for(int i = 0; i < 2; i++)
        output.push_back(Perceptron(6));
        
        
    //kreiranje primera za ucenje
    //40 primera za ucenja
    srand(time(NULL));

    vector<vector<double> > x;
    vector<vector<int> > y;

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
            
            vector<int> pom2 = vector<int>(2);
            if(yp == 0)
            {
               pom2[0] = 1;
               pom2[1] = 0;
            }
            else
            {
               pom2[0] = 0;
               pom2[1] = 1;
            }
                
            y.push_back(pom2);    
    }
        
    uci(x, y);
    printf("Uspesno nauceno\n\n");
    
    //kreiranje test primera i racunanje rezultat mreze
    vector<double>test;
    test.push_back(1.0);
    test.push_back(1.0);
    test.push_back(1.0);
    
    sracunaj(test);
        
        
    system("pause");
    return EXIT_SUCCESS;
}
