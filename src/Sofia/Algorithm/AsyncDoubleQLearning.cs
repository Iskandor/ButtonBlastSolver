using FLAB;
using System;

namespace Sofia
{
    public class AsyncDoubleQLearning : AsyncQLearning
    {
        public AsyncDoubleQLearning(Optimizer p_optimizer, NeuralNetwork p_networkQ, NeuralNetwork p_networkQt, float p_gamma, int p_asynchUpdate) : base(p_optimizer, p_networkQ, p_networkQt, p_gamma, p_asynchUpdate) {

        }

        override protected float CalcMaxQa(Vector p_state)
        {
            _networkQ.Activate(p_state);
            _networkQt.Activate(p_state);

            int maxa = 0;

            for (int i = 0; i < _networkQ.Output.Size; i++)
            {
                if (_networkQ.Output[i] > _networkQ.Output[maxa])
                {
                    maxa = i;
                }
            }

            return _networkQt.Output[maxa];
        }
    }
}
