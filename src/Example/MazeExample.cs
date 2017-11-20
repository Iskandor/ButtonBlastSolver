using FLAB;
using Sofia;
using Sofia.Algorithm.Exploration;
using System;
using System.Collections.Generic;

public class MazeExample
{
    DQN _agent;
    IExploration _exp;
    Vector _state0, _state1;
    int _moves;
    int _wins, _loses;
    bool _firstRun;
    double _accReward;

    FrozenLake _environment;

    bool waitingForInterface;
    bool started;
    private long _timestamp;

    Logger _logger;

    public bool Started
    {
        get { return started; }
    }

    public DQN Agent
    {
        get { return _agent; }
    }

    public void Start(IExploration p_exp)
    {
        _logger = new Logger();
        _timestamp = DateTime.Now.ToFileTime();
#if !DEBUG
        _logger.Init(".\\app_" + _timestamp + ".log");
#endif

        _wins = _loses = 0;
        _exp = p_exp;
        _agent = new DQN();
        _firstRun = true;
        _environment = new FrozenLake();
        Reset();
    }

    public void Update()
    {
        if (started && waitingForInterface == false)
        {
            int action = _agent.ChooseAction(_exp, _agent.GetEstimate(_state0));
            waitingForInterface = true; // musi byt este pred send
            _environment.DoAction(action);
            OnGridClickCompleted();
        }
    }

    public void UpdateParams(int p_e, int p_epochs)
    {
        _exp.UpdateParams((float)p_e / p_epochs);
        Console.WriteLine("Exploration param >> " + _exp.ToString());
    }

    public void Reset()
    {
        _moves = 0;
        _accReward = 0;
        _environment.Reset();
        waitingForInterface = true; // musi byt este pred send
        OnStartGameCompleted();
        started = true;
    }

    public void Quit()
    {
        _agent.Dispose();
        _logger.Close();
    }

    private void OnStartGameCompleted()
    {
        if (_firstRun)
        {
            _agent.SetParam(BaseAgent.STATE_DIM, 4 * FrozenLake.DIM_X * FrozenLake.DIM_Y);
            _agent.SetParam(BaseAgent.ACTION_DIM, 4);
            //_solver.Load();
            _agent.Init();
            _firstRun = false;
        }

        _moves = 1;
        Vector.Release(_state0);
        _state0 = EncodeState(_environment.GetState());
        waitingForInterface = false;
    }

    private void OnGridClickCompleted()
    {
        Vector.Release(_state1);
        _state1 = EncodeState(_environment.GetState());
        float reward = _environment.GetReward();
        int finished = _environment.IsFinished();

        _accReward += reward;

        _agent.Train(_state0, _state1, reward, finished != 0);

        if (finished != 0)
        {
            if (finished > 0) _wins++;
            if (finished < 0) _loses++;

            _logger.Log(_wins.ToString());
            Console.WriteLine((_wins + _loses) + " : " + _moves + " , " + _wins + " / " + _loses + " : " + _accReward);
            started = false;
        }

        _moves++;
        Vector.Release(_state0);
        _state0 = Vector.Copy(_state1);

        waitingForInterface = false;
    }

    private Vector EncodeState(int[,] p_data)
    {
        List<float> tmp = new List<float>();

        foreach (int s in p_data)
        {
            tmp.AddRange(EncodeCell(s));
        }

        Vector result = Vector.Build(_agent.GetParam(BaseAgent.STATE_DIM), tmp);

        return result;
    }

    private float[] EncodeCell(int p_cell)
    {
        float[] result = new float[4];

        if (p_cell > 0)
        {
            result[p_cell - 1] = 1;
        }

        return result;
    }
}
