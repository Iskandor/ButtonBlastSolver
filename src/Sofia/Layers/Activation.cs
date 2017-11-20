using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Sofia
{
    class Activation
    {
        public static double Linear(double x)
        {
            return x;
        }

        public static double Binary(double x)
        {
            return x > 0 ? 1 : 0;
        }

        public static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x)); ;
        }

        public static double Tanh(double x)
        {
            return Math.Tanh(x);
        }

        public static double Softplus(double x)
        {
            return Math.Log(1 + Math.Exp(x));
        }

        public static double ReLU(double x)
        {
            return Math.Max(0d, x);
        }

        public static double dSigmoid(double x)
        {
            return x * (1 - x); ;
        }

        public static double dTanh(double x)
        {
            return 1 - Math.Pow(x, 2);
        }

        public static double dSoftplus(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        public static double dReLU(double x)
        {
            return (x > 0) ? 1 : 0;
        }
    }
}
