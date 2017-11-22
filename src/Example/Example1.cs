using FLAB;
using Sofia.Algorithm.Exploration;
using System;

namespace ButtonBlastSolver
{
    public class Example1
    {
        public Example1()
        {
            SolverConfig.GetInstance().Load(@".\config.maze.json");
        }

        public void Run()
        {
            int EPOCHS = SolverConfig.GetInstance().epochs;

            Console.WriteLine("Epochs: " + EPOCHS);

            MazeExample tester = new MazeExample();

            //BasePool.StrongControl = true;

            IExploration exp = new EGreedyExploration(SolverConfig.GetInstance().epsilon, 0f);
            exp.Init(0.02f, 0f);
            //IExploration exp = new BoltzmannExploration(0.12f, 0.06f);
            tester.Start(exp);

            for (int e = 0; e < EPOCHS; e++)
            {
                while (tester.Started)
                {
                    tester.Update();
                }
                tester.Reset();
                tester.UpdateParams(e, EPOCHS);
                //BasePool.Instance.Check();
            }

            BasePool.Instance.Check();
            tester.Quit();
        }
    }
}
