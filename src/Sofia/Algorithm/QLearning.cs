using FLAB;
using System;

namespace Sofia
{
    public class QLearning
    {
        private NeuralNetwork _network;
        private Optimizer _optimizer;
        private float _gamma;

        public QLearning(Optimizer p_optimizer, NeuralNetwork p_network, float p_gamma) {
            _optimizer = p_optimizer;
            _network = p_network;
            _gamma = p_gamma;
        }

        public void Dispose()
        {
            _optimizer.Dispose();
        }

        public double Train(Vector p_state0, int p_action0, Vector p_state1, float p_reward, bool final = false)
        {
            float mse = 0;
            float maxQs1a = CalcMaxQa(p_state1);

            // updating phase for Q(s,a)
            _network.Activate(p_state0);

            Vector target = _network.Output;

            if (final)
            {
                target[p_action0] = p_reward;
            }
            else
            {
                target[p_action0] = p_reward + _gamma * maxQs1a;
            }

            mse = _optimizer.Train(p_state0, target);

            return mse;
        }

        private float CalcMaxQa(Vector p_state)
        {
            _network.Activate(p_state);

            float maxQa = _network.Output[0];

            for (int i = 0; i < _network.Output.Size; i++)
            {
                if (_network.Output[i] > maxQa)
                {
                    maxQa = _network.Output[i];
                }
            }

            return maxQa;
        }

        public void SetAlpha(float p_alpha)
        {
            if (_optimizer != null)
            {
                _optimizer.Alpha = p_alpha;
            }
        }

        public void Activate(Vector p_input)
        {
            _network.Activate(p_input);
        }

        public Vector Output
        {
            get { return _network.Output; }
        }

        public NeuralNetwork Network
        {
            get { return _network; }
        }
    }
}
