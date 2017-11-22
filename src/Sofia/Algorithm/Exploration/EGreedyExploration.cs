using FLAB;

namespace Sofia.Algorithm.Exploration
{
    class EGreedyExploration : IExploration
    {
        private float _startE, _endE;
        private float _epsilon;
        private float _lowerLimit;
        private float _sigma;

        public EGreedyExploration(float p_startE, float p_endE)
        {
            _epsilon = _startE = p_startE;
            _endE = p_endE;
        }

        public void Init(params float[] p_params)
        {
            _lowerLimit = p_params[0];
            _sigma = p_params[1];
        }

        public int ChooseAction(Vector p_estimates, bool p_probabilities = false)
        {
            int action = 0;
            double random = RandomGenerator.getInstance().Rand(0f, 1f);

            if (random < _epsilon)
            {
                action = RandomGenerator.getInstance().Rand(0, p_estimates.Size - 1);
            }
            else
            {
                int maxi = 0;

                for (int i = 0; i < p_estimates.Size; i++)
                {
                    if (p_estimates[maxi] < p_estimates[i])
                    {
                        maxi = i;
                    }
                }

                action = maxi;
            }

            return action;
        }

        public void UpdateParams(float p_f)
        {
            if (_epsilon > _lowerLimit)
            {
                _epsilon = _startE + (_endE - _startE) * p_f;
            }

            if (_sigma > 0)
            {
                _epsilon = (_epsilon + RandomGenerator.getInstance().Normal(_epsilon, _sigma) / 10);                
            }

            if (_epsilon < _lowerLimit)
            {
                _epsilon = _lowerLimit;
            }
        }

        public float Epsilon
        {
            get { return _epsilon; }
        }

        override public string ToString()
        {
            return _epsilon.ToString();
        }
    }
}
