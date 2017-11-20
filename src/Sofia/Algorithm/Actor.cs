
using FLAB;

namespace Sofia { 

    public class Actor
    {
        private NeuralNetwork _network;
        private Optimizer _optimizer;
        private float _gamma;

        public Actor(Optimizer p_optimizer, NeuralNetwork p_network, float p_gamma)
        {
            _optimizer = p_optimizer;
            _network = p_network;
            _gamma = p_gamma;
        }

        public double Train(Vector p_state0, int p_action, float p_value0, float p_value1, float p_reward)
        {
            double mse = 0;

            _network.Activate(p_state0);
            Vector target = Vector.Copy(_network.Output);
            target[p_action] = p_reward + _gamma * p_value1 - p_value0;

            mse = _optimizer.Train(p_state0, target);

            return mse;
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
