using FLAB;
using System;

namespace ButtonBlastSolver
{
    public class Experiment2
    {
        private SolverManager _solver;

        public Experiment2()
        {
            SolverConfig.GetInstance().Load(@".\config.json");
            _solver = new SolverManager();
        }

        public void Run()
        {
            _solver.Init(__Run);
        }

        private void __Run()
        {
            int EPOCHS = SolverConfig.GetInstance().epochs;

            Console.WriteLine("Parallelism: " + SolverConfig.GetInstance().async_learners);
            Console.WriteLine("Level: " + SolverConfig.GetInstance().level);
            Console.WriteLine("Epochs: " + EPOCHS);
            Console.WriteLine("Epoch " + _solver.FG);

            _solver.Start();

            while (_solver.FG < EPOCHS)
            {
                _solver.Update(EPOCHS);
            }

            BasePool.Instance.Check();
            _solver.Quit();
        }
    }
}
