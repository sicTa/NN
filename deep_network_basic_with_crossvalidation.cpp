#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <algorithm>

double ITA = 1e-2;

int brSkrivenih = 5;
int brUlaz = 2;
int brIzlaz = 2;
using namespace std;

//klasa
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
               return suma + bias;
        }


public:
       //rezultat izracnavanja
       //A je skalarni proizvod w i x(vektor aparametara)
       //Z je rezultat sigmoidne funkcije
       vector<double> w;

       //izvod sigmoidne funkcije
       int izvod;
       //
       double delta;

       //rezultat racunanja citavog enurona
       double Z;

       //bias
       double bias;
       Perceptron(int n):n(n)
       {
               w = vector<double>(n);
               for(int i = 0; i < n; i++)
                       w[i] = (double)(rand()%10) / 10;
               bias = (double)(rand()%10 - 5) / 10;
       }

       //aktivaciona funkcija prvog dela output neurona
       double linear_sigmoid(vector<double> x)
       {
              return  skalarni_proizvod(w, x);
       }
       //aktivaciona funkcija skrivenog neurona
       double ReLU(vector<double> x)
       {
              double proizvod = skalarni_proizvod(w, x);
              if(proizvod <= 0)
              {
                 izvod = 0;
                 return 0;
              }
              else
              {
                 izvod = 1;
                 return proizvod;
              }
       }

};

//aktivaciona funkcija drugog dela output neurona
double softmax(vector<double> x, int j)
{
    double sum = 0.0;
    for (int i = 0; i < x.size(); i++)
        sum+=exp(x[i]);
    return exp(x[j])/sum;
}


vector<Perceptron> hidden_1;
vector<Perceptron> hidden_2;
vector<Perceptron> output;


void uci(vector<vector<double> > x, vector<vector<int> > y)
{
    for(int brojac = 0; brojac < 400; brojac++)
     {
        for(int p = 0; p < x.size(); p++)
             {

                    //propagacija unapred
                    vector<double> z;
                    for(int i = 0; i < brSkrivenih; i++)
                    {
                        z.push_back(hidden_1[i].ReLU(x[p]));
                        hidden_1[i].Z = z[i];
                    }

                    vector<double>z2;
                    for(int i = 0; i < brSkrivenih; i++)
                    {
                         z2.push_back(hidden_2[i].ReLU(z));
                         hidden_2[i].Z = z2[i];
                    }

                    vector<double>out;
                    for(int i = 0; i < brIzlaz; i++)
                        out.push_back(output[i].linear_sigmoid(z2));
                    for(int i = 0; i < brIzlaz; i++)
                        output[i].Z = softmax(out, i);


                    //propagacija unazad
                    //izracunavanje gradijenta greske svih neurona
                    //output
                    for(int i = 0; i < brIzlaz; i++)
                         output[i].delta = y[p][i] - output[i].Z;

                    //hidden 2
                    for(int i = 0; i < brSkrivenih; i++)
                    {
                        double suma = 0.0;
                        for(int j = 0; j < brIzlaz; j++)
                        {
                            suma+=output[j].delta * output[j].w[i];
                        }
                        hidden_2[i].delta = suma *  hidden_2[i].izvod;
                    }

                    //hidden 1
                    for(int i = 0; i < brSkrivenih; i++)
                    {
                        double suma = 0.0;
                        for(int j = 0; j < brSkrivenih; j++)
                        {
                            suma+=hidden_2[j].delta * hidden_2[j].w[i];
                        }
                        hidden_1[i].delta = suma *  hidden_1[i].izvod;
                    }



                    //menjanje vektora tezina
                    //hidden 1
                    for(int i = 0; i < brSkrivenih; i++)
                    {
                         for(int j = 0; j < hidden_1[i].w.size(); j++)
                        {
                            double trenutno = ITA * hidden_1[i].delta * x[p][j];
                            hidden_1[i].w[j] += trenutno;
                        }
                        hidden_1[i].bias += ITA * hidden_1[i].delta;
                    }

                    //hidden2
                    for(int i = 0; i < brSkrivenih; i++)
                    {
                         for(int j = 0; j < hidden_2[i].w.size(); j++)
                        {
                            double trenutno = ITA * hidden_2[i].delta * hidden_1[i].Z;
                            hidden_2[i].w[j] += trenutno;
                        }
                        hidden_2[i].bias += ITA * hidden_2[i].delta;
                    }





                    //output


                    z.clear();
                    for(int i = 0; i < brSkrivenih; i++)
                    {
                        z.push_back(hidden_1[i].ReLU(x[p]));
                        hidden_1[i].Z = z[i];
                    }

                    z2.clear();
                    for(int i = 0; i < brSkrivenih; i++)
                    {
                         z2.push_back(hidden_2[i].ReLU(z));
                         hidden_2[i].Z = z2[i];
                    }

                    out.clear();
                    for(int i = 0; i < brIzlaz; i++)
                        out.push_back(output[i].linear_sigmoid(z2));
                    for(int i = 0; i < brIzlaz; i++)
                        output[i].Z = softmax(out, i);


                    for(int i = 0; i < brIzlaz; i++)
                    {
                        for(int j = 0; j < output[i].w.size(); j++)
                        {
                            double izvod;
                            if(i == j) izvod = softmax(out, i) * (1 - softmax(out, j));
                            else izvod = softmax(out, i) * (- softmax(out, j));

                            double trenutno = ITA * output[i].delta * izvod * hidden_2[j].Z;
                            output[i].w[j] += trenutno;
                        }
                        //ispostavlja se da je izvod za bias uvek 1 :)
                        output[i].bias += ITA * output[i].delta;
                    }
             }
            ITA *= pow(10, -brojac/500);
     }
}

void crossvalidation(vector<vector<double> > ucenje, vector<vector<int> >y, vector<vector<double> > test_in, vector<vector<int> > test_out)
{
    int true_positive = 0;
    int true_negative = 0;
    int false_positive = 0;
    int false_negative = 0;
    hidden_1.clear();
    hidden_2.clear();
    output.clear();
    for(int i = 0; i < brSkrivenih; i++)
        hidden_1.push_back(Perceptron(brUlaz));
    for(int i = 0; i < brSkrivenih; i++)
        hidden_2.push_back(Perceptron(brSkrivenih));
    for(int i = 0; i < brIzlaz; i++)
        output.push_back(Perceptron(brSkrivenih));

    uci(ucenje, y);

            //racunanje
            for(int poz = 0; poz < test_in.size(); poz++)
            {
                 vector<double> z;
                    for(int i = 0; i < brSkrivenih; i++)
                    {
                        z.push_back(hidden_1[i].ReLU(test_in[poz]));
                        hidden_1[i].Z = z[i];
                    }

                    vector<double>z2;
                    for(int i = 0; i < brSkrivenih; i++)
                    {
                         z2.push_back(hidden_2[i].ReLU(z));
                         hidden_2[i].Z = z2[i];
                    }

                    vector<double>out;
                    for(int i = 0; i < brIzlaz; i++)
                        out.push_back(output[i].linear_sigmoid(z2));


                    printf("Izlaz je:  ");
                    for(int i = 0; i < brIzlaz; i++)
                    {
                         output[i].Z = softmax(out, i);
                         printf("%.2llf    ", output[i].Z);
                    }
                    printf("\n");
            }


}


int main()
{


    //kreiranje primera za ucenje
    //40 primera za ucenja
    vector<vector<double> > x;
    vector<vector<int> > y;

    for(int i = 0; i < 100; i++)
    {
            int x1 = rand()%2;
            int x2 = rand()%2;
            int yp  = x1 ^ x2;

            double x1p = (double)x1 + (double)(rand()%10) / 100;
            double x2p = (double)x2 + (double)(rand()%10) / 100;

            vector<double> pom = vector<double>(3);
            pom[0] = x1p;
            pom[1] = x2p;
            x.push_back(pom);

            vector<int> pom2 = vector<int>(2);
            if(x1 == x2)
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
    //uci(x, y);


/*
    vector<vector<double> > t;
    vector<double> pom;
    pom.push_back(1.0);
    pom.push_back(1.0);
    t.push_back(pom);
    pom.clear();

    pom.push_back(1.0);
    pom.push_back(0.0);
    t.push_back(pom);
    pom.clear();

    pom.push_back(0.0);
    pom.push_back(1.0);
    t.push_back(pom);
    pom.clear();

    pom.push_back(0.0);
    pom.push_back(0.0);
    t.push_back(pom);
    pom.clear();


    */


/*
    //cilj desetostruke kros validacije jeste da
    for(int i = 0; i < 10; i++)
    {
        printf("rezultati za test primer %d su: \n", i + 1);
        vector<vector<double> > ucenje;
        vector<vector<int> > izlaz;
        vector<vector<double> > test_in;
        vector<vector<int> > test_out;

        ///cepanje vektora treniranja
        int poc = i * 10;
        int kraj = (i + 1) * 10;
        for(int j = 0; j < poc; j++)
        {
            ucenje.push_back(x[j]);
            izlaz.push_back(y[j]);
        }
        for(int j = poc; j < kraj; j++)
        {
            test_in.push_back(x[j]);
            printf("%d     %d\n", y[j][0], y[j][1]);
            test_out.push_back(y[j]);
        }
        for(int j = kraj; j < 100; j++)
        {
            ucenje.push_back(x[j]);
            izlaz.push_back(y[j]);
        }
        crossvalidation(ucenje, izlaz, test_in, test_out);
        printf("***********************\n");
    }
    printf("*****************");
    printf("\n");


*/





    system("pause");
    return 0;
}
