using FLAB;
using Sofia.Algorithm.Exploration;
using Sofia.Layers;
using System;
using System.Collections.Generic;

namespace Sofia
{
    public class AsyncQN : BaseAgent
    {
        private List<AsyncDoubleQLearning> _learners;

        private int _action;
        private int _learner, _learnerSize;
        private int _qtUpdateIndex, _qtUpdateSize;

        private NeuralNetwork _networkQ;
        private NeuralNetwork _networkQt;

        public AsyncQN(int p_learners)
        {
            _learnerSize = p_learners;
            _learners = new List<AsyncDoubleQLearning>(p_learners);
        }

        override public void Init(NeuralNetwork p_network = null)
        {
            _qtUpdateIndex = 0;
            _qtUpdateSize = SolverConfig.GetInstance().qtupdate_size;
            _networkQ = null;

            if (p_network == null)
            {
                _networkQ = new NeuralNetwork();
                _networkQ.AddLayer("input", new InputLayer(GetParam(STATE_DIM)), BaseLayer.TYPE.INPUT);
                _networkQ.AddLayer("hidden0", new CoreLayer(SolverConfig.GetInstance().hidden_layer, ACTIVATION.RELU, BaseLayer.TYPE.HIDDEN), BaseLayer.TYPE.HIDDEN);
                _networkQ.AddLayer("output", new CoreLayer(GetParam(ACTION_DIM), ACTIVATION.LINEAR, BaseLayer.TYPE.OUTPUT), BaseLayer.TYPE.OUTPUT);

                // feed-forward connections
                _networkQ.AddConnection("input", "hidden0", Connection.INIT.GLOROT_UNIFORM);
                _networkQ.AddConnection("hidden0", "output", Connection.INIT.GLOROT_UNIFORM);
            }
            else
            {
                _networkQ = p_network;
            }

            CreateNetworkQt();

            for (int i = 0; i < _learners.Capacity; i++)
            {
                AsyncDoubleQLearning worker = new AsyncDoubleQLearning(new ADAM(_networkQ), _networkQ, _networkQt, 0.99f, SolverConfig.GetInstance().async_update);
                //worker.SetAlpha(SolverConfig.GetInstance().learning_rate);
                worker.Optimizer.InitAlpha(SolverConfig.GetInstance().learning_rate, SolverConfig.GetInstance().learning_rate / 10);
                _learners.Add(worker);
            }
        }

        override public void Dispose()
        {            
        }

        private void CreateNetworkQt()
        {
            Console.WriteLine("Initializing networkQt started");
            _networkQt = IOUtils.LoadNetwork(IOUtils.SaveNetwork(_networkQ));
            Console.WriteLine("Initializing networkQt finished");
        }

        public void SetLearner(int p_value)
        {
            _learner = p_value;
        }

        override public Vector GetEstimate(Vector p_state0)
        {
            return _learners[_learner].Activate(p_state0);
        }

        override public int ChooseAction(IExploration p_exp, Vector p_estimate)
        {
            _action = p_exp.ChooseAction(p_estimate);

            /*
            BoltzmannExploration exp = (BoltzmannExploration)p_exp;

            for(int i = 0; i < 5; i++)
            {
                exp.Temperature = 0.5f * i + 0.15f; 
                exp.ChooseAction(p_estimate);
            }
            */

            return _action;
        }

        override public void Train(Vector p_state0, Vector p_state1, float p_reward, bool p_finished)
        {
            _learners[_learner].Train(p_state0, _action, p_state1, p_reward, p_finished);
        }

        public void UpdateQt()
        {
            _qtUpdateIndex++;
            if (_qtUpdateIndex == _qtUpdateSize)
            {
                _qtUpdateIndex = 0;
                _networkQt.OverrideParams(_networkQ);
            }
        }

        public void ResetContext()
        {
            _networkQ.ResetContext();
            _networkQt.ResetContext();
        }

        override public Optimizer GetOptimizer()
        {
            return _learners[_learner].Optimizer;
        }

        override public void Save(string p_filename)
        {
            string data = null;

            data = IOUtils.SaveNetwork(_networkQ);
            System.IO.File.WriteAllText(@".\" + p_filename, data);
        }

        override public void Load(string p_filename)
        {
            NeuralNetwork network = new NeuralNetwork();
            string data = null;

            data = System.IO.File.ReadAllText(@".\" + p_filename);
            network = IOUtils.LoadNetwork(data);

            Init(network);
        }
    }
}
