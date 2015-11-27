using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace PerceptronLib
{
    public class SekvencaUcenja
    {
        public double []x; //ulazni parametri
        public double y;//rezultat
    }
    public class Perceptron
    {
        SekvencaUcenja  []s;
        Random r = new Random();
        double[] w; //vektor tezina
        int n;//duzina vektora tezina
        public Perceptron()
        {
            w = new double[1];
            w[0] = 1E-5;
        }
        public Perceptron(int n)
        {
            w = new double[n];
            for (int i = 0; i < n; i++) w[i] = r.Next(0, 2) / 1000;
        }

        /// <summary>
        /// Primarni konstruktor!
        /// </summary>
        /// <param name="_s">sekvenca ucenja koja se ubacuje</param>
        public Perceptron(SekvencaUcenja [] _s)
        {
            s = _s;
            n = _s[0].x.Length;
            w = new double[n];
            for (int i = 0; i < n; i++) w[i] = r.Next(0, 2) / 1000;
        }

        /// <summary>
        /// vektori x i y moraju biti iste duzine!!!!
        /// </summary>
        /// <param name="x">prvi vektor</param>
        /// <param name="y">drugi vektor</param>
        /// <returns>skalarni proizovd prvog i drugog vektora</returns>
        double skalarniProizvod(double[] x, double[] y)
        {
            double rez = 0;
            for (int i = 0; i < x.Length; i++) rez += x[i] * y[i];
            return rez;
        }

        double norma(double[] x)
        {
            return Math.Sqrt(skalarniProizvod(x, x));
        }
        /// <summary>
        /// algoritam ispravki tezina
        /// </summary>
        /// <param name="w">pocetni vektor tezina</param>
        /// <param name="gama">granicna vrednost koju uslov
        /// ucenja mora da zadovolji (treshold)</param>
        /// <param name="t">broj iteracija u slucaju da se zadata granicna
        /// vrednost ne moze postici</param>
        /// <param name="trenutni">trenutna iteracija</param>
        /// <returns>vraca ispravljen vektor tezina</returns>
        double[] ucenjeLogicticki(double[] w, double gama, int t, int trenutni)
        {
            if (trenutni <= t)
            {
                double[] wi = new double[w.Length];
                double[] pom = new double[w.Length];

                for (int i = 0; i < w.Length; i++) pom[i] = 0;

                for (int i = 0; i < s.Length; i++)
                {
                    double exponent = Math.Pow(Math.E, -1 * skalarniProizvod(w, s[i].x));
                    double exponentF = 1 - exponent;
                    double proizvod = (s[i].y + 1) * (Math.Pow(exponentF, -2) + Math.Pow(exponentF, -3)) * exponent;
                    for (int j = 0; j < s[i].x.Length; j++)
                        pom[j] += s[i].x[j] * proizvod * gama * 2;
                }

                if (norma(pom) < gama) return w;

                for (int i = 0; i < w.Length; i++)
                    wi[i] = w[i] - pom[i];

                return ucenjeLogicticki(wi, gama, t, trenutni + 1);
            }

            else return w;
        }

    }
}
