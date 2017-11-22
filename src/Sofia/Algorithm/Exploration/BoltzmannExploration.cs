using FLAB;
using System;

namespace Sofia.Algorithm.Exploration
{
    class BoltzmannExploration : IExploration
    {
        private float _startTemp, _endTemp;
        private float _temperature;
        private float _minTemp;

        public BoltzmannExploration(float p_startTemp, float p_endTemp)
        {
            _startTemp = p_startTemp;
            _endTemp = p_endTemp;
            _temperature = _startTemp;
            _minTemp = 0f;
        }

        public void Init(params float[] p_params)
        {
            _minTemp = p_params[0];
        }

        public int ChooseAction(Vector p_estimates, bool p_probabilities = false)
        {
            int dim = p_estimates.Size;
            float[] pi = new float[dim];
            float[] exp = new float[dim];
            float sumexp = 0;
            int action = 0;

            if (p_probabilities)
            {
                for (int i = 0; i < dim; i++)
                {
                    pi[i] = p_estimates[i];
                }
            }
            else
            {
                for (int i = 0; i < dim; i++)
                {
                    exp[i] = (float)Math.Exp(p_estimates[i] / _temperature);
                    sumexp += exp[i];
                }

                for (int i = 0; i < dim; i++)
                {
                    pi[i] = exp[i] / sumexp;
                }
            }

            /*
            for(int i = 0; i < 4; i++)
            {
                Console.Write(pi[i] + " ");
            }
            Console.WriteLine();
            */

            action = RandomGenerator.getInstance().GetRandom(pi);

            if (action == -1) action = 0;

            return action;
        }

        public void UpdateParams(float p_f)
        {
            if (_temperature > _minTemp)
            {
                _temperature = _startTemp + (_endTemp - _startTemp) * p_f;
            }
            else
            {
                _temperature = _minTemp;
            }
            
        }

        public float Temperature
        {
            get { return _temperature; }
            set { _temperature = value; }
        }

        override public string ToString()
        {
            return _temperature.ToString();
        }
    }
}
