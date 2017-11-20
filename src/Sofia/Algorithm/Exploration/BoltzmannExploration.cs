using FLAB;
using System;

namespace Sofia.Algorithm.Exploration
{
    class BoltzmannExploration : IExploration
    {
        private float _startExp, _endExp;
        private float _temperature;

        public BoltzmannExploration(float p_startExp, float p_endExp)
        {
            _startExp = p_startExp;
            _endExp = p_endExp;
            _temperature = _startExp;
        }

        public void Init(params float[] p_params)
        {
        }

        public int ChooseAction(Vector p_estimates)
        {
            int dim = p_estimates.Size;
            float[] pi = new float[dim];
            float[] exp = new float[dim];
            float sumexp = 0;
            int action = 0;

            for (int i = 0; i < dim; i++)
            {
                exp[i] = (float)Math.Exp(p_estimates[i] / _temperature);
                sumexp += exp[i];
            }

            for (int i = 0; i < dim; i++)
            {
                pi[i] = exp[i] / sumexp;
            }

            action = RandomGenerator.getInstance().GetRandom(pi);

            if (action == -1) action = 0;

            return action;
        }

        public void UpdateParams(float p_f)
        {
            //float exp = _startExp + (_endExp - _startExp) * p_f;
            _temperature = p_f < 0.5f ? _startExp : _endExp;
        }

        public float Temperature
        {
            get { return _temperature; }
        }

        override public string ToString()
        {
            return _temperature.ToString();
        }
    }
}
