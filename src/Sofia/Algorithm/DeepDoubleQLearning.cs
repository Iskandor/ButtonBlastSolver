using FLAB;

namespace Sofia
{
    public class DeepDoubleQLearning : DeepQLearning
    {
        public DeepDoubleQLearning(Optimizer p_optimizer, NeuralNetwork p_network, float p_gamma, int p_capacity, int p_batchSize, int p_qtUpdateSize) : base(p_optimizer, p_network, p_gamma, p_capacity, p_batchSize, p_qtUpdateSize)
        {
        }

        override protected float CalcMaxQa(Vector p_state)
        {
            _Qnetwork.Activate(p_state);
            _QTnetwork.Activate(p_state);

            int maxa = 0;

            for (int i = 0; i < _Qnetwork.Output.Size; i++)
            {
                if (_Qnetwork.Output[i] > _Qnetwork.Output[maxa])
                {
                    maxa = i;
                }
            }

            return _QTnetwork.Output[maxa];
        }
    }
}