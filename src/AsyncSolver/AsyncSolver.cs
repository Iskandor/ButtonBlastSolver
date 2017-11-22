using FLAB;
using Sofia;
using Sofia.Algorithm.Exploration;
using System;
using System.Collections.Generic;

public class AsyncSolver : Solver
{
    private int _learner;
    private GameGoalDto[] _goals0;

    private Match3LevelConfig _levelConfig;

    public AsyncSolver(InputInterface p_inputInterface, StateEncoder p_encoder, BaseAgent p_agent, IExploration p_exp, Logger p_logger) : base(p_inputInterface, p_encoder, p_agent, p_exp, p_logger)
    {
    }

    public void Start(int p_learner, bool p_test = false)
    {
        _learner = p_learner;
        _test = p_test;
        _wins = _loses = 0;
        Reset();
    }

    override protected void OnStartGameCompleted(StartGameRequest.ResultData result)
    {
        _levelConfig = result.levelConfig;
        CreateGoals(result.gameGoals.Count);
        CopyGoals(result.gameGoals);

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
        /*
        _gridCheck = 0;
        GridCheck();
        */
        waitingForInterface = false;
    }

    override public void Update()
    {
        if (started && waitingForInterface == false)
        {
            ((AsyncQN)_agent).SetLearner(_learner);
            _action = _agent.ChooseAction(_exploration, _agent.GetEstimate(_state0));
            int row = _action / 9; //Random.Range(0, 8);
            int column = _action % 9; //Random.Range(0, 8);
            waitingForInterface = true; // musi byt este pred send

            //Console.WriteLine(_learner + " click ::" + _gameState.levelScore);

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

    override protected void OnGridClickCompleted(GridClickRequest.ResultData result)
    {
        //Console.WriteLine(_learner + " response ::" + result.levelScore);

        if (result.validTurn)
        {
            _validMoves++;
            int row = _action / 9;
            int column = _action % 9;
            //_DEBUG_clicks.Add(new Vector(3, new List<float> { 0f, row, column }));
        }
        _moves++;

        Train(result.validTurn, result.gridData, result.gameGoals, result.powerupScore);

        boost = false;

        _gameState.grid = result.gridData;
        _gameState.goals = result.gameGoals;
        _gameState.randomHistory = result.randomHistory;
        _gameState.powerupScore = result.powerupScore;
        _gameState.levelScore = result.levelScore;

        if (result.validTurn)
        {
            CopyGoals(result.gameGoals);
        }
        waitingForInterface = false;
    }

    override protected void Train(bool p_validTurn, List<GridDto> p_grid, List<GameGoalDto> p_goals, int p_powerupScore)
    {
        Vector.Release(_state1);
        _state1 = _encoder.EncodeState(p_grid, p_powerupScore, _agent.GetParam(BaseAgent.STATE_DIM), boost);
        float reward = GetReward(p_validTurn, p_goals);
        _accReward += reward;
        int finished = IsFinished(p_validTurn, p_goals);

        if (!_test)
        {            
            _agent.Train(_state0, _state1, reward, finished != 0);
        }

        if (finished != 0)
        {
            if (_validMoves > _maxMoves)
            {
                _loses++;
            }
            else
            {
                _wins++;
            }
            Console.WriteLine(_learner + " >> moves (" + _validMoves + " / " + _moves + ") , win rate (" + _wins + " / " + _loses + ") : reward " + _accReward);
            _logger.Log(_wins + ";" + _loses + ";" + _seed + ";" + _validMoves + ";" + _moves + ";" + _accReward);
            started = false;
        }

        Vector.Release(_state0);
        _state0 = Vector.Copy(_state1);
    }

    override protected float GetReward(bool p_validTurn, List<GameGoalDto> p_goals)
    {
        return Reward1(p_validTurn, p_goals);
    }

    private float Reward1(bool p_validTurn, List<GameGoalDto> p_goals)
    {
        float reward = 0;

        if (p_validTurn)
        {
            int finished = IsFinished(true, p_goals);

            if (finished == 1)
            {
                reward = 10;
            }
        }
        else
        {
            reward = -1;
        }

        return reward;
    }

    private float Reward2(bool p_validTurn, List<GameGoalDto> p_goals)
    {
        float reward = 0;

        if (p_validTurn)
        {
            bool diff = false;

            for (int i = 0; i < p_goals.Count; i++)
            {
                if (_goals0[i].remainingCount - p_goals[i].remainingCount > 0)
                {
                    diff = true;
                }
            }

            if (diff)
            {
                int sum = 0;
                float advantage = 0;

                for (int i = 0; i < p_goals.Count; i++)
                {
                    sum++;
                    advantage += (float)(_levelConfig.goals[i].count - p_goals[i].remainingCount) / _levelConfig.goals[i].count;
                }

                advantage /= sum;
                reward = advantage * 10;
            }
        }
        else
        {
            reward = -1;
        }

        return reward;
    }

    private float Reward3(bool p_validTurn, List<GameGoalDto> p_goals)
    {
        float reward = 0;

        if (p_validTurn)
        {
            int finished = IsFinished(true, p_goals);

            if (finished == 1)
            {
                reward = 10;
            }
            else
            {
                double diff = 0;

                for (int i = 0; i < p_goals.Count; i++)
                {
                    diff += Math.Pow((float)(_goals0[i].remainingCount - p_goals[i].remainingCount) / _levelConfig.goals[i].count, 2);
                }

                diff = Math.Sqrt(diff);

                if (diff > 0)
                {
                    double advantage = 0;

                    for (int i = 0; i < p_goals.Count; i++)
                    {
                        advantage += Math.Pow(1 - ((float)(_levelConfig.goals[i].count - p_goals[i].remainingCount) / _levelConfig.goals[i].count), 2);
                    }

                    advantage = Math.Sqrt(advantage);

                    //Console.WriteLine(advantage);
                    //Console.WriteLine(diff);

                    reward = (float)Math.Exp(-advantage);
                    //Console.WriteLine(reward);
                }
            }
        }
        else
        {
            reward = -1;
        }

        return reward;
    }

    private void CreateGoals(int p_size)
    {
        _goals0 = new GameGoalDto[p_size];

        for(int i = 0; i < _goals0.Length; i++)
        {
            _goals0[i] = new GameGoalDto();
        }
    }

    private void CopyGoals(List<GameGoalDto> p_goals)
    {
        for(int i = 0; i < p_goals.Count; i++)
        {
            _goals0[i].index = p_goals[i].index;
            _goals0[i].remainingCount = p_goals[i].remainingCount;
        }
    }
}