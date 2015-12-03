using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Perceptron
{
    public class TrainingSeq
    {
        public int x, y;
        public double rez;
        public TrainingSeq(int _x, int _y, int _rez)
        {
            x = _x;
            y = _y;
            rez = _rez;
        }
    }

    public class Perceptron
    {
        public double w1, w2;
        public Perceptron()
        {
            w1 = w2 = 0;
        }


        /// <summary>
        /// f*ck thic piece of code in particular 
        /// </summary>
        /// <param name="trenutni"></param>
        /// <param name="ukupno"></param>
        /// <param name="s"></param>
        //public void uci(int trenutni, int ukupno, TrainingSeq[] s)
        //{
        //    if (trenutni < ukupno)
        //    {
        //        double pom1, pom2;
        //        pom1 = pom2 = 0;
        //        double suma = 0;
        //        for (int i = 0; i < s.Length; i++)
        //        {
        //            suma = -1.0*(w1 * s[i].x + w2 * s[i].y);
        //            double exponent = Math.Exp(suma);
        //            double expF = Math.Pow(1 + exponent, -1);
        //            double p = Math.Pow(expF, 2);
        //            double q = Math.Pow(expF, 3);
        //            pom1 += (s[i].rez + 1) * (p + q) * exponent * s[i].x / s.Length;
        //            pom2 += (s[i].rez + 1) * (p + q) * exponent * s[i].y / s.Length;
        //            suma = 0;
        //        }
        //        w1 -= 2 * 0.001 * pom1;
        //        w2 -= 2 * 0.001 * pom2;
        //        pom1 = pom2 = 0;

        //        //NE KORISTITI OVO
        //        //NIKAAAAAAAADDDD!!!
        //        //JASGDKHJASGXDKHASCVLASCVFASYHFcv
        //    }
        //}




        public void uci2(int trenutno, int ukupno, TrainingSeq[] s)
        {
            double greska = 1;
            while (greska > 0.0001)
            {
                double d1, d2;
                d1 = d2 = 0;
                for (int i = 0; i < s.Length; i++)
                {
                    double rezi = 1 / (1 + Math.Exp(-1 * (w1 * s[i].x + w2 * s[i].y)));
                    greska = s[i].rez - rezi;
                    d1 += greska * s[i].x;
                    d2 += greska * s[i].y;
                }
                w1 = w1 + 0.0001 * d1;
                w2 = w2 + 0.0001 * d2;
            }

        }

        public int rezultat(int x, int y)
        {
            double rezi = 1 / (1 + Math.Exp(-1 * (w1 * x + w2 * y)));
            if (rezi > 0.5) return 1;
            else return 0;
        }
    }
    class Program
    {
       
        static void Main(string[] args)
        {
            Random r = new Random();
            TrainingSeq[] s = new TrainingSeq[1000];
            for (int i = 0; i < 1000; i++)
            {
                int x = r.Next(0, 10);
                int y = r.Next(0, 10);
                while (y == x) y = r.Next(0, 10);
                int rez;
                if (x > y) rez = 1; else rez = 0;
                s[i] = new TrainingSeq(x, y, rez);
            }
            Perceptron p = new Perceptron();
            p.uci2(1, 100, s);
            Console.WriteLine("Tezina w1: " + p.w1);
            Console.WriteLine("Tezina w1: " + p.w2);
            Console.WriteLine("Pritisnite bilo koje dugme");
            Console.ReadLine();
            int brojac = 0;
            for (int i = 0; i < 100; i++)
            {
                int x = r.Next(0, 10);
                int y = r.Next(0, 10);
                while (y == x) y = r.Next(0, 100);
                int control;
                if (x > y) control = 1; else control = 0;
                int vecemanje = p.rezultat(x, y);
                if (control == vecemanje)
                    brojac++;
                Console.WriteLine("X: {0}   Y: {1}  X>Y: {2}", x, y, vecemanje);
            }

            Console.WriteLine("Procentualno pogodjeno: " + brojac + "%");

            
            Console.ReadLine();
        }
    }
}
