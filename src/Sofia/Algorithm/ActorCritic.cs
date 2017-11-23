
using FLAB;

namespace Sofia { 

    public class ActorCritic
    {
        protected NeuralNetwork _network;
        protected Optimizer _optimizer;
        protected float _gamma;

        public ActorCritic(Optimizer p_optimizer, NeuralNetwork p_network, float p_gamma)
        {
            _optimizer = p_optimizer;
            _network = p_network;
            _gamma = p_gamma;
        }

        public void Dispose()
        {
            //_optimizer.Dispose();
        }

        virtual public double Train(Vector p_state0, int p_action, Vector p_state1, float p_reward, bool p_final)
        {
            double mse = 0;
            int criticOutput = _network.Output.Size - 1;

            _network.Activate(p_state1);
            float value1 = _network.Output[criticOutput];
            _network.Activate(p_state0);
            float value0 = _network.Output[criticOutput];

            Vector target = Vector.Copy(_network.Output);
            target[p_action] = p_reward + _gamma * value1 - value0;
            target[criticOutput] = p_reward + _gamma * value1;

            mse = _optimizer.Train(p_state0, target);

            Vector.Release(target);

            return mse;
        }

        virtual public void SetAlpha(float p_alpha)
        {
            if (_optimizer != null)
            {
                _optimizer.Alpha = p_alpha;
            }
        }

        public Vector Activate(Vector p_input)
        {
            return _network.Activate(p_input);
        }

        public Vector Output
        {
            get { return _network.Output; }
        }

        public NeuralNetwork Network
        {
            get { return _network; }
        }

        public Optimizer Optimizer
        {
            get { return _optimizer; }
        }
    }

}
