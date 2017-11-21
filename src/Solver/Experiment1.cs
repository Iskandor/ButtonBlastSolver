using FLAB;
using Newtonsoft.Json;
using Sofia;
using Sofia.Algorithm.Exploration;
using System;
using System.Collections.Generic;
using System.IO;

namespace ButtonBlastSolver
{
    public class Experiment1
    {
        private InputInterface _inputInterface;
        private StateEncoder _encoder;

        private DDQN _agent;
        private long _timestamp;
        private Logger _logger;

        public int FG;

        private Action _initialized;

        private Solver _solver;
        private bool _tester;

        public Experiment1()
        {
            SolverConfig.GetInstance().Load(@".\config.json");
            _timestamp = DateTime.Now.ToFileTime();
            _logger = new Logger();
#if !DEBUG
            _logger.Init(".\\app_" + _timestamp + ".log");
#endif
            _logger.Log(JsonConvert.SerializeObject(SolverConfig.GetInstance()));

            int asyncLearners = SolverConfig.GetInstance().async_learners;

            _inputInterface = new InputInterface();
            _encoder = new StateEncoder(_inputInterface);
            _agent = new DDQN();
            IExploration exp = new EGreedyExploration(SolverConfig.GetInstance().epsilon, 0f);
            exp.Init(0.1f, 1f);


            _solver = new Solver(_inputInterface, _encoder, _agent, exp, _logger);
        }

        public void Run(bool p_tester = false)
        {
            Init(__Run, p_tester);
        }

        private void Init(Action p_initialized, bool p_tester)
        {
            _tester = p_tester;
            _initialized = p_initialized;
            int seed = SolverConfig.GetInstance().seed;

            /*
            if (seed == 0)
            {
                seed = RandomGenerator.getInstance().rand(1, 5);
            }
            */

            (new StartGameRequest(_inputInterface)).Send(SolverConfig.GetInstance().level, seed, OnStartGameCompleted);
        }

        private void OnStartGameCompleted(StartGameRequest.ResultData result)
        {
            _encoder.PrepareEncoderProjection(result.levelConfig);
            _agent.SetParam(BaseAgent.ACTION_DIM, _encoder.GetActionDim());
            _agent.SetParam(BaseAgent.BOOST_INDEX, _encoder.GetActionDim() - 1);
            _agent.SetParam(BaseAgent.STATE_DIM, _encoder.GetStateDim());
            Console.WriteLine("Max. moves: " + result.levelConfig.moves);
            Console.WriteLine("Action dim: " + _agent.GetParam(BaseAgent.ACTION_DIM));
            Console.WriteLine("State  dim: " + _agent.GetParam(BaseAgent.STATE_DIM));

            if (SolverConfig.GetInstance().filename != string.Empty)
            {
                string filename = ".\\" + SolverConfig.GetInstance().filename;
                Console.WriteLine("Loading file " + filename);
                _agent.Load(filename);
                Console.WriteLine(filename + " loaded.");
            }
            else
            {
                _agent.Init();
            }

            _initialized.Invoke();
        }

        private void __Run()
        {
            int EPOCHS = SolverConfig.GetInstance().epochs;

            Console.WriteLine("Level: " + SolverConfig.GetInstance().level);
            Console.WriteLine("Epochs: " + EPOCHS);

            IExploration exp = new EGreedyExploration(SolverConfig.GetInstance().epsilon, 0f);
            exp.Init(0f, 0f);
            //tester.Start(new BoltzmannExploration(2, 0, -4));

            _solver.Start(_tester);
            //BasePool.StrongControl = true;

            for (int e = 0; e < EPOCHS; e++)
            {
                while (_solver.Started)
                {
                    _solver.Update();
                }
                _solver.Reset();
                if (!_tester) _solver.UpdateParams(e, EPOCHS);
                Console.WriteLine(exp.ToString());
                //BasePool.Instance.Check();
            }

            BasePool.Instance.Check();
            Quit();
        }

        private void Quit()
        {
            _logger.Close();
            Console.WriteLine("Saving...");
            if (_tester)
            {
                SaveHistogram(_solver.Histogram);
            }
            else
            {
                _agent.Save("critic_" + _timestamp + ".json");
            }
            
            Console.WriteLine("Saved");
        }

        private void SaveHistogram(Dictionary<int, int> p_histogram)
        {
            StreamWriter file = new StreamWriter("hist_" + _timestamp + ".csv", false);

            foreach(KeyValuePair<int, int> p in p_histogram)
            {
                file.WriteLine(p.Key + ";" + p.Value);
            }

            file.Close();
        }
    }
}
