using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace Perception
{
    public partial class Form1 : Form
    {
        public class Neuron
        {
            struct SekvencaUcenja
            {
                public double [] x1;
                public double [] rezultat;
            }


            //promenljive
            Random r = new Random();
            private double output;
            private double[] x;
            private double[] w;
            private double eta = 0.000001;

            //osnovni konstruktor
            public Neuron() { }

            //konstruktor ciji je argiment vektor x, 
            //gde je x feature vektor
            public Neuron(double[] _x)
            {
                x = _x;
                for (int i = 0; i < _x.Length; i++)
                {
                    w[i] = r.Next(0, 101) / 100;
                }
            }

            double[] ucenje(SekvencaUcenja[] S, int trenutni, int t, double [] w)
            {
                if (trenutni == 0) return w;
                else if (trenutni <= t)
                {
                    double[] w1 = new double[w.Length];
                    double[] pom = new double[w.Length];
                    for (int i = 0; i < w.Length; i++)
                        pom[i] = 0;

                    //w1 = w + 2*eta*suma od 0 do duzine sekvence ucenja od 
                    //vrednosti rezultata proizvoda x1 sekvence ucenja i razlike rezultata
                    //sekvence i proizvoda tezina i x1


                    for (int i = 0; i < w.Length; i++)
                    {     
                        for (int k = 0; i < S.Length; i++)
                        {
                            for (int j = 0; j < w.Length; j++)
                            {
                                pom[j] += pomnoziNizove(S[k].x1,
                                    oduzmiNizove(S[k].rezultat,
                                    pomnoziNizove(S[k].x1, w)))[j];
                            }
                        }
                        w1[i] = w[i] + 2 * eta * pom[i];
                    }


                    return ucenje(S, trenutni + 1, t, w1);
                }
                else return w;    
            }



            double[] pomnoziNizove(double[] a, double[] b)
            {
                double[] rez = new double[a.Length];
                for (int i = 0; i < a.Length; i++)
                    rez[i] = a[i] * b[i];
                return rez;
            }

            double[] oduzmiNizove(double[] a, double[] b)
            {
                double[] rez = new double[a.Length];
                for (int i = 0; i < a.Length; i++)
                    rez[i] = a[i] - b[i];
                return rez;
            }

        }
        public Form1()
        {
            InitializeComponent();
        }
    }
}
