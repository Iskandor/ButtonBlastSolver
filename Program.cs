using cudaFLAB;
using FLAB;
using Sofia;
using Sofia.Algorithm.Exploration;
using Sofia.Layers;
using System;
using System.Diagnostics;

namespace ButtonBlastSolver
{
    class Program
    {

        static void Main(string[] args)
        {
            /*
            XOR xor = new XOR();
            xor.Run();
            */

            Example1 example1 = new Example1();
            example1.Run();

            /*
            Experiment1 experiment1 = new Experiment1();
            experiment1.Run(true);
            */

            /*
            Experiment2 experiment2 = new Experiment2();
            experiment2.Run();
            */

            Console.WriteLine("Press any key...");
            Console.ReadLine();
        }
    }
}
