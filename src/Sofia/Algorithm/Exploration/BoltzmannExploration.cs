using FLAB;
using System;

namespace Sofia.Algorithm.Exploration
{
    class BoltzmannExploration : IExploration
    {
        private float _startTemp, _endTemp;
        private float _temperature;

        public BoltzmannExploration(float p_startTemp, float p_endTemp)
        {
            _startTemp = p_startTemp;
            _endTemp = p_endTemp;
            _temperature = _startTemp;
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
            _temperature = _startTemp + (_endTemp - _startTemp) * p_f;
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
