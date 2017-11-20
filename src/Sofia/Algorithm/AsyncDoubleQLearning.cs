using FLAB;
using System;

namespace Sofia
{
    public class AsyncDoubleQLearning
    {
        private NeuralNetwork _networkQ;
        private NeuralNetwork _networkQt;

        private NeuralNetwork _networkQa;
        private NeuralNetwork _networkQb;
        private Optimizer _optimizer;
        private float _gamma;

        private AsyncUnit _asyncUnit;

        public AsyncDoubleQLearning(Optimizer p_optimizer, NeuralNetwork p_networkQa, NeuralNetwork p_networkQb, float p_gamma, int p_asynchUpdate) {
            _optimizer = p_optimizer;
            _networkQa = p_networkQa;
            _networkQb = p_networkQb;
            _gamma = p_gamma;

            _asyncUnit = new AsyncUnit();
            _asyncUnit.Init(p_asynchUpdate);
            _optimizer.InitAsynchMode(true);
        }

        public void Dispose()
        {
            //_optimizer.Dispose();
        }

        public double Train(Vector p_state0, int p_action0, Vector p_state1, float p_reward, bool p_final = false)
        {
            ChooseNetwork();

            float mse = 0;
            float maxQs1a = CalcMaxQa(p_state1);

            // updating phase for Q(s,a)
            _networkQ.Activate(p_state0);

            Vector target = Vector.Copy(_networkQ.Output);

            if (p_final)
            {
                target[p_action0] = p_reward;
            }
            else
            {
                target[p_action0] = p_reward + _gamma * maxQs1a;
            }

            mse = _optimizer.Train(p_state0, target);
            Vector.Release(target);

            _asyncUnit.Update(p_final);

            if (_asyncUnit.IsAsyncReady)
            {
                _optimizer.AsyncUpdate();
            }

            return mse;
        }

        private float CalcMaxQa(Vector p_state)
        {
            _networkQt.Activate(p_state);

            float maxQa = _networkQt.Output[0];

            for (int i = 0; i < _networkQt.Output.Size; i++)
            {
                if (_networkQt.Output[i] > maxQa)
                {
                    maxQa = _networkQt.Output[i];
                }
            }

            return maxQa;
        }

        private void ChooseNetwork()
        {
            if (RandomGenerator.getInstance().Rand(0, 1) == 0)
            {
                _networkQ = _networkQa;
                _networkQt = _networkQb;
            }
            else
            {
                _networkQ = _networkQb;
                _networkQt = _networkQa;
            }
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
            _networkQ.Activate(p_input);
        }

        public Vector Output
        {
            get { return _networkQ.Output; }
        }

        public NeuralNetwork Network
        {
            get { return _networkQ; }
        }
    }
}
