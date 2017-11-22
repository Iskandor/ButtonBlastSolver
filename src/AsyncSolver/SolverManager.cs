using Newtonsoft.Json;
using Sofia;
using Sofia.Algorithm.Exploration;
using System;
using System.Collections.Generic;

public class SolverManager
{
    private InputInterface _inputInterface;
    private StateEncoder _encoder;
    
    private AsyncQN _agent;
    private AsyncSolver[] _learners;
    private Dictionary<int, int> _actorMonitor;

    private long _timestamp;
    private Logger _logger;

    public int FG;

    private Action _initialized;

    public SolverManager()
    {
        FG = 0;
        _timestamp = DateTime.Now.ToFileTime();
        _logger = new Logger();
#if !DEBUG
        _logger.Init(".\\app_" + _timestamp + ".log");
#endif
        _logger.Log(JsonConvert.SerializeObject(SolverConfig.GetInstance()));

        int asyncLearners = SolverConfig.GetInstance().async_learners;
        _actorMonitor = new Dictionary<int, int>();

        _inputInterface = new InputInterface();
        _encoder = new StateEncoder(_inputInterface);
        _agent = new AsyncQN(asyncLearners);
        _learners = new AsyncSolver[asyncLearners];        

        for (int i = 0; i < asyncLearners; i++)
        {
            IExploration exp = null;
            exp = new BoltzmannExploration(SolverConfig.GetInstance().epsilon, 0.15f);
            exp.Init(0.15f);
            //exp = new EGreedyExploration(SolverConfig.GetInstance().epsilon, 0f);
            //exp.Init(0.05f, 0f);

            /*
            if (i % 2 == 0)
            {
                exp = new EGreedyExploration(SolverConfig.GetInstance().epsilon, 0f);
                exp.Init(0.02f, 0f);
            }
            else
            {
                exp = new BoltzmannExploration(0.12f, 0.12f);
            }
            */

            _learners[i] = new AsyncSolver(_inputInterface, _encoder, _agent, exp, _logger);
        }
    }

    public void Init(Action p_initialized)
    {
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

    public void Start()
    {
        for(int i = 0; i < SolverConfig.GetInstance().async_learners; i++)
        {
            _learners[i].Start(i);
        }
    }

    public void Update(int p_epochs)
    {
        for (int i = 0; i < SolverConfig.GetInstance().async_learners; i++)
        {
            if (_learners[i].Started)
            {
                _learners[i].Update();
            }
            else
            {                
                FG++;
                _learners[i].Reset();
                _learners[i].UpdateParams(FG, p_epochs / 2);


                if (_actorMonitor.ContainsKey(i))
                {
                    _actorMonitor[i]++;
                }
                else
                {
                    _actorMonitor[i] = 1;
                }

                foreach (int v in _actorMonitor.Values)
                {
                    Console.Write(v + " ");
                }
                Console.WriteLine();
                Console.WriteLine("Epoch " + FG);
            }
            _agent.UpdateQt();
        }
    }

    public void Quit()
    {
        _logger.Close();
        Console.WriteLine("Saving...");
        _agent.Save("critic_" + _timestamp + ".json");
        Console.WriteLine("Saved");
    }
}