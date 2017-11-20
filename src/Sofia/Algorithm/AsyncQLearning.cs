using FLAB;
using System;

namespace Sofia
{
    public class AsyncQLearning
    {
        private NeuralNetwork _networkQ;
        private NeuralNetwork _networkQt;
        private Optimizer _optimizer;
        private float _gamma;

        private AsyncUnit _asyncUnit;

        public AsyncQLearning(Optimizer p_optimizer, NeuralNetwork p_networkQ, NeuralNetwork p_networkQt, float p_gamma, int p_asynchUpdate) {
            _optimizer = p_optimizer;
            _networkQ = p_networkQ;
            _networkQt = p_networkQt;
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

        public void SetAlpha(float p_alpha)
        {
            if (_optimizer != null)
            {
                _optimizer.Alpha = p_alpha;
            }
        }

        public Optimizer Optimizer
        {
            get { return _optimizer; }
        }

        public Vector Activate(Vector p_input)
        {
            return _networkQ.Activate(p_input);
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
