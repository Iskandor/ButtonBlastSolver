using FLAB;
using Newtonsoft.Json;
using Sofia;
using Sofia.Algorithm.Exploration;
using System;
using System.Collections.Generic;

// ----------------- EXAMPLE ------------------
public class Solver
{

    protected struct StateVar
    {
        public List<GridDto>                    grid;
        public List<GameGoalDto>                goals;
        public List<GridGenerator.HistoryDto>   randomHistory;
        public int levelScore;
        public int powerupScore;
    }

    protected InputInterface  _inputInterface;
    protected StateEncoder _encoder;
    protected IExploration _exploration;
    protected BaseAgent _agent;
    protected Logger _logger;

    protected  StateVar _gameState;
    protected  Vector _state0, _state1;
    protected int _action;
    protected int _validMoves, _moves, _maxMoves;
    protected int _wins, _loses;
    protected bool _firstRun;
    protected int _level;
    protected int _seed;
    protected double _accReward;
    protected bool _test;
    protected int LIMIT;

    protected bool waitingForInterface;
    protected bool started;
    protected bool boost;

    private Logger _DEBUG_logger;
    private List<Vector> _DEBUG_clicks;
    private int _DEBUG_limit;

    private Dictionary<int, int> _histogram;

    public bool Started
    {
        get { return started; }
    }

    public Solver(InputInterface p_inputInterface, StateEncoder p_encoder, BaseAgent p_agent, IExploration p_exp, Logger p_logger)
    {
        _inputInterface = p_inputInterface;
        _encoder = p_encoder;
        _agent = p_agent;
        _exploration = p_exp;
        _logger = p_logger;
        _gameState = new StateVar();
        _histogram = new Dictionary<int, int>();
    }

    public void Start(bool p_test = false)
    {
        _test = p_test;
        _wins = _loses = 0;

        _DEBUG_logger = new Logger();
        _DEBUG_limit = SolverConfig.GetInstance().batch_size * 15;       
        Reset();
    }

    virtual public void Update()
    {
        if (started && waitingForInterface == false)
        {
            _action = _agent.ChooseAction(_exploration, _agent.GetEstimate(_state0));
            int row = _action / 9; //Random.Range(0, 8);
            int column = _action % 9; //Random.Range(0, 8);
            waitingForInterface = true; // musi byt este pred send

            if (_action == _agent.GetParam(BaseAgent.BOOST_INDEX))
            {
                OnPowerupSelect();
            }
            else
            {
                new GridClickRequest(_inputInterface).Send(_level, _seed, _gameState.levelScore, _gameState.powerupScore, _gameState.goals, _gameState.grid, _gameState.randomHistory, row, column, boost, OnGridClickCompleted);
            }
        }
    }

    public void UpdateParams(int p_e, int p_epochs)
    {
        _exploration.UpdateParams((float)p_e / (p_epochs / 2));
        Console.WriteLine("Exploration param >> " + _exploration.ToString());
        _agent.GetOptimizer().UpdateAlpha((float)p_e / p_epochs);
        Console.WriteLine("Learning param >> " + _agent.GetOptimizer().Alpha.ToString());
    }

    public void Reset()
    {
        _accReward = 0;
        waitingForInterface = true; // musi byt este pred send

        _level = SolverConfig.GetInstance().level;
        _seed = SolverConfig.GetInstance().seed;

        if (_seed == 0)
        {
            _seed = RandomGenerator.getInstance().Rand(1, 5);
        }

        (new StartGameRequest(_inputInterface)).Send(SolverConfig.GetInstance().level, _seed, OnStartGameCompleted);

        _DEBUG_clicks = new List<Vector>();

        started = true;
    }

    virtual protected void OnStartGameCompleted(StartGameRequest.ResultData result)
    {        
        _moves = 1;
        _validMoves = 1;
        _maxMoves = result.levelConfig.moves;
        LIMIT = _maxMoves * 10;
        Vector.Release(_state0);
        _state0 = _encoder.EncodeState(result.gridData, 0, _agent.GetParam(BaseAgent.STATE_DIM), boost);
        _gameState.grid = result.gridData;
        _gameState.goals = result.gameGoals;
        _gameState.randomHistory = null;
        _gameState.powerupScore = 0;
        _gameState.levelScore = 0;
        boost = false;
        waitingForInterface = false;
    }

    protected void OnPowerupSelect()
    {
        boost = true;
        Train(true, _gameState.grid, _gameState.goals, _gameState.powerupScore);
        waitingForInterface = false;
    }

    virtual protected void OnGridClickCompleted(GridClickRequest.ResultData result)
    {
        if (result.validTurn)
        {
            _validMoves++;
            int row = _action / 9;
            int column = _action % 9;
            //Console.WriteLine("click " + row + "," + column);
            //_DEBUG_clicks.Add(new Vector(3, new List<float> { 0f, row, column }));
        }
        _moves++;
        _gameState.grid = result.gridData;
        _gameState.goals = result.gameGoals;
        _gameState.randomHistory = result.randomHistory;
        _gameState.powerupScore = result.powerupScore;
        _gameState.levelScore = result.levelScore;

        boost = false;
        Train(result.validTurn, result.gridData, result.gameGoals, result.powerupScore);
        waitingForInterface = false;
    }

    virtual protected void Train(bool p_validTurn, List<GridDto> p_grid, List<GameGoalDto> p_goals, int p_powerupScore)
    {
        Vector.Release(_state1);
        _state1 = _encoder.EncodeState(p_grid, p_powerupScore, _agent.GetParam(BaseAgent.STATE_DIM), boost);
        float reward = GetReward(p_validTurn, p_goals);
        _accReward += reward;
        int finished = IsFinished(p_validTurn, p_goals);

        if (!_test) {
            _agent.Train(_state0, _state1, reward, finished != 0);
        }

        if (finished != 0)
        {
            //if (finished > 0) _wins++;
            //if (finished < 0) _loses++;
            if (_test)
            {
                if (_histogram.ContainsKey(_validMoves))
                {
                    _histogram[_validMoves]++;
                }
                else
                {
                    _histogram[_validMoves] = 1;
                }
            }
            
            Console.WriteLine("Epoch " + (_wins + _loses) + ", seed " + _seed + " >> moves (" + _validMoves + " / " + _moves + ") , win rate (" + _wins + " / " + _loses + ") : reward " + _accReward);
            _logger.Log(_wins + ";" + _loses + ";" + _seed + ";" + _validMoves + ";" + _moves + ";" + _accReward);
            started = false;
        }

        /*
        if (_moves > _DEBUG_limit && _agent.Epsilon > 0.5d)
        {
            DEBUG_LogError();
            started = false;
        }
        */

        Vector.Release(_state0);
        _state0 = Vector.Copy(_state1);
    }

    protected void DEBUG_LogError()
    {
        long timestamp = DateTime.Now.ToFileTime();
        _DEBUG_logger.Init(".\\error_" + timestamp + ".log");

        foreach (Vector v in _DEBUG_clicks)
        {
            if (v[0] == 0f)
            {
                _DEBUG_logger.Log("Click " + v[1] + "," + v[2]);
            }
            if (v[0] == 1f)
            {
                _DEBUG_logger.Log("Powerup " + v[1] + "," + v[2]);
            }            
        }
        _DEBUG_logger.Close();
    }

    protected int IsFinished(bool p_validTurn, List<GameGoalDto> p_goals)
    {
        bool goalsCompleted = true;
        bool turnsDepleted = _moves == LIMIT; //_validMoves == _maxMoves; podmienka ukoncenia hry

        foreach (GameGoalDto g in p_goals)
        {
            goalsCompleted = goalsCompleted && (g.remainingCount == 0);
            //Console.WriteLine(g.index + " " + g.remainingCount);
        }

        int result = 0;

        if (turnsDepleted)
        {
            result = -1;
        }
        if (goalsCompleted)
        {
            result = 1;
        }

        return result;
    }

    virtual protected float GetReward(bool p_validTurn, List<GameGoalDto> p_goals)
    {
        float reward = 0;

        if (p_validTurn)
        {
            int finished = IsFinished(true, p_goals); // ignorujeme nevalidne kola

            if (finished == 1)
            {
                if (_validMoves > _maxMoves)
                {
                    _loses++;
                    reward = 1;
                }
                else
                {
                    _wins++;
                    reward = 10;
                }                
            }
        }
        else
        {
            reward = -1;
        }

        return reward;
    }

    public Dictionary<int, int> Histogram
    {
        get { return _histogram; }
    }
}