using System;
using System.Collections.Generic;
using FLAB;
using Sofia.Algorithm.Exploration;

namespace Sofia
{
    public class BaseAgent
    {
        public const string BOOST_INDEX = "BOOST_INDEX";
        public const string ACTION_DIM = "ACTION_DIM";
        public const string STATE_DIM = "STATE_DIM";

        private Dictionary<string, int> _params;

        public BaseAgent()
        {
            _params = new Dictionary<string, int>();
        }

        virtual public int ChooseAction(IExploration p_exp, Vector p_state0)
        {
            return 0;
        }

        virtual public void Train(Vector p_state0, Vector p_state1, float p_reward, bool p_finished)
        {
        }

        virtual public void Dispose()
        {
        }

        virtual public void Init(NeuralNetwork p_network = null)
        {
        }

        virtual public void Load(string p_filename)
        {
        }

        virtual public void Save(string p_filename)
        {
        }

        public void SetParam(string p_param, int p_value)
        {
            _params[p_param] = p_value;
        }

        public int GetParam(string p_param)
        {
            return _params[p_param];
        }

        virtual public Vector GetEstimate(Vector p_state0)
        {
            return null;
        }

        virtual public Optimizer GetOptimizer()
        {
            return null;
        }
    }
}
